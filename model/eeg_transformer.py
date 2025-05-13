import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import lightning as pl
import math

from torchmetrics import Accuracy, ConfusionMatrix, F1Score, MetricCollection, Precision, Recall

debug_mode_flag = False

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        n_odd = d_model // 2
        pe[:, 0, 1::2] = torch.cos(position * div_term[:n_odd])
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EEGNet(nn.Module):
    def __init__(self, F1=16, eegnet_kernel_size=32, D=2, eeg_chans=22, eegnet_separable_kernel_size=16,
                 eegnet_pooling_1=8, eegnet_pooling_2=4, dropout=0.5):
        super(EEGNet, self).__init__()

        F2 = F1*D
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
        # x.shape = (B, 1, C, L)
        x = self.block1(x)

        # x.shape = (B, F1, C, L)
        if debug_mode_flag: print('Shape of x after block1 of EEGNet: ', x.shape)

        x = self.bn1(x)
        x = self.block2(x)
        # x.shape = (B, F1*D, 1, L)
        if debug_mode_flag: print('Shape of x after block2 of EEGNet: ', x.shape)

    
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout(x)
        # x.shape = (B, F1*D, 1, L/8)
        if debug_mode_flag: print('Shape of x before block3 of EEGNet: ', x.shape)
        x = self.block3(x)
        # x.shape = (B, F1*D, 1, L/8)
        if debug_mode_flag: print('Shape of x after block3 of EEGNet: ', x.shape)

        x = self.bn3(x)
        x = self.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout(x)
        # x.shape = (B, F1*D, 1, L/64)
        if debug_mode_flag: print('Shape of x by the end of EEGNet: ', x.shape)

        return x

class EEGTransformerNet(nn.Module):
    def __init__(self, nb_classes, sequence_length, eeg_chans=22,
                 F1=16, D=2, eegnet_kernel_size=32, dropout_eegnet=0.3, dropout_positional_encoding=0.3, eegnet_pooling_1=5, eegnet_pooling_2=5, 
                 MSA_num_heads = 8, flag_positional_encoding=True, transformer_dim_feedforward=2048, num_transformer_layers=6):
        super(EEGTransformerNet, self).__init__()
        """
        F1 = the number of temporal filters
        F2 = number of spatial filters
        """

        F2 = F1 * D
        self.sequence_length_transformer = sequence_length//eegnet_pooling_1//eegnet_pooling_2

        self.eegnet = EEGNet(eeg_chans=eeg_chans, F1=F1, eegnet_kernel_size=eegnet_kernel_size, D=D, 
                             eegnet_pooling_1=eegnet_pooling_1, eegnet_pooling_2=eegnet_pooling_2, dropout=dropout_eegnet)
        self.linear = nn.Linear(self.sequence_length_transformer, nb_classes)
        
        self.flag_positional_encoding = flag_positional_encoding
        self.pos_encoder = PositionalEncoding(self.sequence_length_transformer, dropout=dropout_positional_encoding)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.sequence_length_transformer,
            nhead=MSA_num_heads,
            dim_feedforward=transformer_dim_feedforward
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_transformer_layers
        )

    def forward(self, x):
        # TODO switch here dimensions because it is build for a different dataset
        x = torch.permute(x, (0, 2, 1))
        # input x shape: (batch_size, num_channels, seq_len) = (batch_size, 22, 1000)
        x = torch.unsqueeze(x, 1)
        # x = x.permute(0, 2, 3, 1)  # similar to Keras Permute layer
        ## expected input shape for eegnet is (batch_size, 1, num_channels, seq_len)
        # print('Shape of x before EEGNet: ', x.shape)
        x = self.eegnet(x)
        # print('Shape of x after EEGNet: ', x.shape)
        x = torch.squeeze(x) ## output shape is (Batch size, F1*D, L//pool_1//pool2))

        ### Transformer Encoder Module
        x = x.permute(2, 0, 1) # output shape: (seq_len, batch_size, F1*D)
        seq_len_transformer, batch_size_transformer, channels_transformer = x.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.cat((torch.zeros((seq_len_transformer, batch_size_transformer, 1), 
                                   requires_grad=True).to(device), x), 2)
        x = x.permute(2, 1, 0) # ouptut shape: (channels+1, batch_size, seq_len). seq_len is seen as the embedding size
        if debug_mode_flag: print('Shape of x before Transformer: ', x.shape)
        if self.flag_positional_encoding:
            x = x * math.sqrt(self.sequence_length_transformer)
            x = self.pos_encoder(x) ## output matrix shape: (channels+1, batch_size, seq_len)
        if debug_mode_flag: print('Positional Encoding Done!')
        if debug_mode_flag: print('Shape of x after Transformer: ', x.shape)
        # x = self.transformer(x)
        x = self.transformer_encoder(x)  # shape: (channels+1, batch_size, seq_len)
        x = x[0,:,:].reshape(batch_size_transformer, -1) # shape: (batch_size, seq_len)

        ### Linear layer module
        if debug_mode_flag: print('Shape of x before linear layer: ', x.shape)
        x = self.linear(x)
        return x

class EEGGNN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = EEGTransformerNet(
            nb_classes=2,
            sequence_length=self.cfg.sequence_length,
            eeg_chans=self.cfg.num_eeg_channels,
            F1=self.cfg.F1,
            D=self.cfg.D,
            eegnet_kernel_size=self.cfg.eegnet_kernel_size,
            # eegnet_separable_kernel_size=self.cfg.eegnet_separable_kernel_size,
            eegnet_pooling_1=self.cfg.eegnet_pooling_1,
            eegnet_pooling_2=self.cfg.eegnet_pooling_2,
            dropout_eegnet=self.cfg.dropout_eegnet,
            MSA_num_heads=self.cfg.MSA_num_heads,
            transformer_dim_feedforward=self.cfg.transformer_dim_feedforward,
            num_transformer_layers=self.cfg.num_transformer_layers,
            flag_positional_encoding=self.cfg.flag_positional_encoding,
        )
        self.criterion = nn.CrossEntropyLoss()
        metrics = MetricCollection({
            'accuracy':  Accuracy(task="binary", average='none'),
            'accuracy_micro':  Accuracy(task="binary", average='micro'),
            'precision': Precision(task="binary", average='none'),
            'precision_micro': Precision(task="binary", average='micro'),
            'recall':    Recall(task="binary", average='none'),
            'recall_micro':    Recall(task="binary", average='micro'),
            'f1':        F1Score(task="binary", average='none'),
            'f1_micro':        F1Score(task="binary", average='micro'),
            # 'confmat':   ConfusionMatrix(num_classes=2)
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics   = metrics.clone(prefix='val_')
        self.test_metrics  = metrics.clone(prefix='test_')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        preds = logits.argmax(dim=1)
        self.train_metrics.update(preds, y)
        return loss
    
    def on_training_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics)
        self.train_metrics.reset()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        preds = logits.argmax(dim=1)
        self.val_metrics.update(preds, y)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss)
        preds = logits.argmax(dim=1)
        self.test_metrics.update(preds, y)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.cfg.scheduler.step_size,
            gamma=self.cfg.scheduler.gamma
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    # you can do this over a parameter when you create the trainer
    # def clip_gradients( self,
    #     optimizer,
    #     gradient_clip_val,
    #     gradient_clip_algorithm):
    #     # Clip gradients for the main model parameters
    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)


# def define_initial_hyperparameters(eegnet_F1=32, eegnet_D=2, eegnet_kernel_size=32, MSA_num_heads=4):

#     ### model hyperparameters
#     # eegnet_F1, eegnet_D = 64, 2
#     # eegnet_kernel_size = 64
#     MSA_embed_dim = eegnet_F1*eegnet_D + 0
#     # MSA_num_heads = 8

#     dropout = 0.3  # dropout probability

#     model = EEGTransformerNet(nb_classes = nclasses, eeg_chans=num_eeg_channels, sequence_length=sequence_length,
#                     F1=eegnet_F1, D=eegnet_D, eegnet_kernel_size=eegnet_kernel_size, 
#                     MSA_embed_dim = MSA_embed_dim, MSA_num_heads = MSA_num_heads, dropout_eegnet=dropout).to(device)

#     model_stats = summary(model, input_size=(batch_size, num_eeg_channels, sequence_length), verbose=0)
#     print(model_stats)

#     # Define a DataLoader
#     # DataLoader combines a dataset and a sampler, and provides an iterable over the given dataset. It also provides multi-process data loading with the num_workers argument.
    
#     train_dir = os.path.join(data_root, 'train')  # Specify the path to your root directory
#     train_dataset = NumpyDataset(train_dir)
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

#     val_dir = os.path.join(data_root, 'val')  # Specify the path to your root directory
#     val_dataset = NumpyDataset(val_dir)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

#     test_dir = os.path.join(data_root, 'test')  # Specify the path to your root directory
#     test_dataset = NumpyDataset(test_dir)
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

#     lr = 0.0001  # learning rate
#     # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     STEPLR_period = 10.0
#     STEPLR_gamma = 0.1
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, STEPLR_period, gamma=STEPLR_gamma)

#     # best_val_loss = float('inf')
#     epochs = 1000