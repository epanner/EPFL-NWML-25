import os
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, [lowcut, highcut], fs=fs, btype='bandpass', output='sos')
    return sosfiltfilt(sos, data)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EEGNet(nn.Module):
    def __init__(self, F1, D, eeg_chans, eegnet_kernel_size, eegnet_separable_kernel_size,
                 eegnet_pooling_1, eegnet_pooling_2, dropout):
        super().__init__()
        F2 = F1 * D
        self.dropout = nn.Dropout(dropout)
        self.block1 = nn.Conv2d(1, F1, (1, eegnet_kernel_size), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.block2 = nn.Conv2d(F1, F2, (eeg_chans, 1), padding='valid', bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, eegnet_pooling_1))
        self.block3 = nn.Conv2d(F2, F2, (1, eegnet_separable_kernel_size), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avg_pool2 = nn.AvgPool2d((1, eegnet_pooling_2))
    def forward(self, x):
        x = self.elu(self.bn1(self.block1(x)))
        x = self.elu(self.bn2(self.block2(x)))
        x = self.dropout(self.avg_pool1(x))
        x = self.elu(self.bn3(self.block3(x)))
        x = self.dropout(self.avg_pool2(x))
        return x

class EEGTransformerNet(nn.Module):
    def __init__(self, nb_classes, sequence_length, eeg_chans,
                 F1, D, eegnet_kernel_size, eegnet_separable_kernel_size,
                 eegnet_pooling_1, eegnet_pooling_2, dropout_eegnet,
                 MSA_num_heads, transformer_dim_feedforward, num_transformer_layers,
                 flag_positional_encoding=True):
        super().__init__()
        F2 = F1 * D
        seq_len_trans = sequence_length // (eegnet_pooling_1 * eegnet_pooling_2)
        self.eegnet = EEGNet(F1, D, eeg_chans, eegnet_kernel_size,
                             eegnet_separable_kernel_size,
                             eegnet_pooling_1, eegnet_pooling_2,
                             dropout_eegnet)
        self.flag_positional_encoding = flag_positional_encoding
        self.pos_encoder = PositionalEncoding(seq_len_trans, dropout=dropout_eegnet)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=seq_len_trans+1,
                                    nhead=MSA_num_heads,
                                    dim_feedforward=transformer_dim_feedforward),
            num_layers=num_transformer_layers
        )
        self.classifier = nn.Linear(seq_len_trans+1, nb_classes)
        self.sequence_length = sequence_length
        self.eeg_chans = eeg_chans
        self.seq_len_trans = seq_len_trans
    def forward(self, x):  # x: (B, C, L)
        B, C, L = x.size()
        x = x.unsqueeze(1)  # (B,1,C,L)
        x = self.eegnet(x)  # (B,F2,1,L')
        x = x.squeeze(2)  # (B,F2,L')
        x = x.permute(2, 0, 1)  # (L',B,F2)
        zeros = torch.zeros((L//(self.seq_len_trans)*(0)+1, B, 1), device=x.device)
        x = torch.cat((zeros, x), dim=2)  # (L',B,F2+1)
        if self.flag_positional_encoding:
            x = x * math.sqrt(self.seq_len_trans)
            x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (L',B,F2+1)
        out = x[0].reshape(B, -1)
        out = self.classifier(out)
        return out

class NumpyDataset(Dataset):
    def __init__(self, root_dir, classes, sequence_length):
        self.data = []
        self.sequence_length = sequence_length
        for idx, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            for f in os.listdir(cls_dir):
                arr = np.load(os.path.join(cls_dir, f))
                for seg in arr:
                    self.data.append((seg, idx))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)

class TUSZDataModule(pl.LightningDataModule):
    def __init__(self, data_root, segment_interval, batch_size, binary_flag, balanced_flag, seizure_types):
        super().__init__()
        self.data_root = data_root
        self.segment_interval = segment_interval
        self.batch_size = batch_size
        self.binary_flag = binary_flag
        self.balanced_flag = balanced_flag
        self.seizure_types = seizure_types
        self.sequence_length = segment_interval * 250
    def setup(self, stage=None):
        if self.binary_flag:
            classes = ['bckg', 'seizure']
        else:
            classes = ['fnsz', 'gnsz', 'cpsz', 'bckg']
        root = os.path.join(self.data_root, f"segment_interval_{self.segment_interval}_sec")
        self.train_ds = NumpyDataset(os.path.join(root, 'train'), classes, self.sequence_length)
        self.val_ds   = NumpyDataset(os.path.join(root, 'val'),   classes, self.sequence_length)
        self.test_ds  = NumpyDataset(os.path.join(root, 'test'),  classes, self.sequence_length)
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
    def val_dataloader(self):
        return DataLoader(self.val_ds,   batch_size=self.batch_size, shuffle=False, num_workers=4)
    def test_dataloader(self):
        return DataLoader(self.test_ds,  batch_size=self.batch_size, shuffle=False, num_workers=4)

class SeizureDetectionModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = EEGTransformerNet(
            nb_classes = 2 if hparams.binary_flag else 4,
            sequence_length = hparams.sequence_length,
            eeg_chans = hparams.num_eeg_channels,
            F1 = hparams.eegnet_F1,
            D = hparams.eegnet_D,
            eegnet_kernel_size = hparams.eegnet_kernel_size,
            eegnet_separable_kernel_size = hparams.eegnet_kernel_size,
            eegnet_pooling_1 = hparams.eegnet_pooling_1,
            eegnet_pooling_2 = hparams.eegnet_pooling_2,
            dropout_eegnet = hparams.dropout_eegnet,
            MSA_num_heads = hparams.MSA_num_heads,
            transformer_dim_feedforward = hparams.transformer_dim_feedforward,
            num_transformer_layers = hparams.num_transformer_layers
        )
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.STEPLR_period,
            gamma=self.hparams.STEPLR_gamma
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # data and model parameters
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--segment_interval', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--binary_flag', action='store_true')
    parser.add_argument('--balanced_flag', action='store_true')
    parser.add_argument('--sequence_length', type=int, default=1000)
    parser.add_argument('--num_eeg_channels', type=int, default=22)
    parser.add_argument('--eegnet_F1', type=int, default=32)
    parser.add_argument('--eegnet_D', type=int, default=2)
    parser.add_argument('--eegnet_kernel_size', type=int, default=32)
    parser.add_argument('--eegnet_separable_kernel_size', type=int, default=16)
    parser.add_argument('--eegnet_pooling_1', type=int, default=8)
    parser.add_argument('--eegnet_pooling_2', type=int, default=4)
    parser.add_argument('--dropout_eegnet', type=float, default=0.3)
    parser.add_argument('--MSA_num_heads', type=int, default=4)
    parser.add_argument('--transformer_dim_feedforward', type=int, default=2048)
    parser.add_argument('--num_transformer_layers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--STEPLR_period', type=int, default=10)
    parser.add_argument('--STEPLR_gamma', type=float, default=0.1)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=50)
    args = parser.parse_args()

    dm = TUSZDataModule(args.data_root, args.segment_interval, args.batch_size,
                        args.binary_flag, args.balanced_flag, None)
    model = SeizureDetectionModule(args)

    checkpoint_cb = ModelCheckpoint(monitor='val_loss', save_top_k=1)
    early_stop_cb = EarlyStopping(monitor='val_loss', patience=args.patience)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
        gpus=1 if torch.cuda.is_available() else 0
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)
    
Trainer