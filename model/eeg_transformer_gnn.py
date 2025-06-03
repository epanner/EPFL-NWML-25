import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import lightning as pl
import math
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

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


class ECALayer(nn.Module):
    """Efficient Channel Attention.

    Args:
        channels: Number of channels to apply attention over
        gamma: Factor that determines kernel size based on channels
        b: Offset for kernel size calculation
    """

    def __init__(self, channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        t = int(abs(math.log(channels, 2) + b) / gamma) # Increases kernel size adaptively with more channels
        k = t if t % 2 else t + 1 # Ensures kernel size is always odd (clear center)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [batch, channels, height, width]
        b, c, h, w = x.size()

        # Feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Reshape to [batch, 1, channels]
        y = y.reshape(b, 1, c)

        # Two-layer 1D conv for capturing cross-channel interactions
        y = self.conv(y)

        # Reshape to [batch, channels, 1, 1]
        y = y.reshape(b, c, 1, 1)

        # Sigmoid activation for attention weights
        y = self.sigmoid(y)

        # Channel-wise multiplication
        return x * y


class EEGElectrodeGraph:
    """Creates electrode graphs based on distance CSV file"""

    @staticmethod
    def load_distance_csv(csv_path):
        """
        Load electrode distances from CSV file

        Args:
            csv_path: Path to CSV file with columns ['from', 'to', 'distance']

        Returns:
            electrode_names: List of electrode names
            distance_matrix: Symmetric distance matrix
        """
        import pandas as pd

        # Load CSV
        df = pd.read_csv(csv_path)

        # Get unique electrode names
        electrode_names = sorted(list(set(df['from'].unique().tolist() + df['to'].unique().tolist())))
        num_electrodes = len(electrode_names)

        # Create electrode name to index mapping
        name_to_idx = {name: idx for idx, name in enumerate(electrode_names)}

        # Initialize distance matrix
        distance_matrix = torch.full((num_electrodes, num_electrodes), float('inf'))

        # Fill diagonal with zeros (self-distances)
        for i in range(num_electrodes):
            distance_matrix[i, i] = 0.0

        # Fill distance matrix from CSV data
        for _, row in df.iterrows():
            from_idx = name_to_idx[row['from']]
            to_idx = name_to_idx[row['to']]
            distance = float(row['distance'])

            # Make matrix symmetric
            distance_matrix[from_idx, to_idx] = distance
            distance_matrix[to_idx, from_idx] = distance

        return electrode_names, distance_matrix

    @staticmethod
    def create_adjacency_from_distances(distance_matrix, connection_type='knn', k=4, threshold=0.8):
        """
        Create adjacency matrix based on distance matrix

        Args:
            distance_matrix: Tensor of shape (num_electrodes, num_electrodes) with distances
            connection_type: 'knn', 'threshold', or 'adaptive_threshold'
            k: number of nearest neighbors for knn
            threshold: distance threshold for threshold-based connections
        """
        num_electrodes = distance_matrix.shape[0]

        if connection_type == 'knn':
            # Connect each electrode to its k nearest neighbors
            _, indices = torch.topk(distance_matrix, k + 1, largest=False, dim=1)  # +1 to exclude self
            edge_index = []
            for i in range(num_electrodes):
                for j in indices[i][1:]:  # Skip self-connection (distance=0)
                    if distance_matrix[i, j] != float('inf'):  # Only add valid connections
                        edge_index.append([i, j])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()

        elif connection_type == 'threshold':
            # Connect electrodes within threshold distance
            valid_distances = distance_matrix < float('inf')  # Exclude infinite distances
            close_enough = distance_matrix <= threshold
            adj_matrix = valid_distances & close_enough & (distance_matrix > 0)  # Exclude self-connections
            edge_index = adj_matrix.nonzero().t()

        elif connection_type == 'adaptive_threshold':
            # Use median distance as threshold
            valid_distances = distance_matrix[distance_matrix < float('inf')]
            valid_distances = valid_distances[valid_distances > 0]  # Exclude self-connections
            median_distance = torch.median(valid_distances)
            threshold = median_distance.item()

            print(f"Using adaptive threshold: {threshold:.3f}")

            valid_mask = distance_matrix < float('inf')
            close_enough = distance_matrix <= threshold
            adj_matrix = valid_mask & close_enough & (distance_matrix > 0)
            edge_index = adj_matrix.nonzero().t()

        return edge_index


class GraphSpatialBlock(nn.Module):
    """Graph-based spatial processing block to replace conv2d spatial convolution"""

    def __init__(self, in_channels, out_channels, distance_csv_path=None,
                 electrode_names=None, distance_matrix=None, heads=4, dropout=0.1,
                 connection_type='knn', k=4, threshold=0.8):
        super(GraphSpatialBlock, self).__init__()
        self.heads = heads

        # Graph Attention Network
        self.gat = GATConv(
            in_channels,
            out_channels // heads,
            heads=heads,
            dropout=dropout,
            concat=True
        )

        # Create electrode graph from distance data
        if distance_csv_path is not None:
            self.electrode_names, distance_matrix = EEGElectrodeGraph.load_distance_csv(distance_csv_path)
            self.num_electrodes = len(self.electrode_names)
        elif distance_matrix is not None and electrode_names is not None:
            self.electrode_names = electrode_names
            self.num_electrodes = len(electrode_names)
            distance_matrix = distance_matrix
        else:
            raise ValueError("Either distance_csv_path or both distance_matrix and electrode_names must be provided")

        # Create adjacency matrix from distances
        self.edge_index = EEGElectrodeGraph.create_adjacency_from_distances(
            distance_matrix, connection_type=connection_type, k=k, threshold=threshold
        )

        print(f"Created graph with {self.num_electrodes} electrodes and {self.edge_index.shape[1]} edges")
        print(f"Electrodes: {self.electrode_names}")

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, num_electrodes, time_steps)
        Returns:
            Output tensor of shape (batch_size, out_channels, 1, time_steps)
        """
        batch_size, in_channels, num_electrodes, time_steps = x.shape

        # Move edge_index to same device as input
        edge_index = self.edge_index.to(x.device)

        # Reshape for graph processing: (batch_size * time_steps, num_electrodes, in_channels)
        x = x.permute(0, 3, 2, 1).reshape(-1, num_electrodes, in_channels)

        # Apply graph attention for each time-batch combination
        graph_outputs = []
        for i in range(x.shape[0]):  # For each batch-time combination
            # Extract features for this time point: (num_electrodes, in_channels)
            node_features = x[i]

            # Apply graph attention
            graph_out = self.gat(node_features, edge_index)  # (num_electrodes, out_channels)

            # Global pooling across electrodes (spatial aggregation)
            graph_out = torch.mean(graph_out, dim=0, keepdim=True)  # (1, out_channels)
            graph_outputs.append(graph_out)

        # Stack outputs: (batch_size * time_steps, 1, out_channels)
        x = torch.stack(graph_outputs, dim=0)

        # Reshape back to: (batch_size, time_steps, 1, out_channels)
        x = x.reshape(batch_size, time_steps, 1, -1)

        # Permute to: (batch_size, out_channels, 1, time_steps)
        x = x.permute(0, 3, 2, 1)

        # Apply batch normalization across the channel dimension
        # Reshape for batch norm: (batch_size, out_channels, time_steps)
        x_bn = x.squeeze(2)  # Remove the singleton dimension
        x_bn = self.bn(x_bn)
        x = x_bn.unsqueeze(2)  # Add back the singleton dimension

        return x


class EEGNetGNN(nn.Module):
    """EEGNet with Graph Neural Network integration for spatial processing"""

    def __init__(self, F1=16, eegnet_kernel_size=32, D=2, distance_csv_path=None,
                 electrode_names=None, distance_matrix=None,
                 eegnet_separable_kernel_size=16, eegnet_pooling_1=8, eegnet_pooling_2=4,
                 dropout=0.5, use_eca=False, gnn_heads=4, gnn_dropout=0.1,
                 connection_type='knn', gnn_k=4, gnn_threshold=0.8):
        super(EEGNetGNN, self).__init__()

        F2 = F1 * D
        self.dropout = nn.Dropout(dropout)

        # Block 1: Temporal convolution (unchanged)
        self.block1 = nn.Conv2d(1, F1, (1, eegnet_kernel_size), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2: Spatial processing with GNN (REPLACEMENT)
        self.spatial_gnn = GraphSpatialBlock(
            in_channels=F1,
            out_channels=F2,
            distance_csv_path=distance_csv_path,
            electrode_names=electrode_names,
            distance_matrix=distance_matrix,
            heads=gnn_heads,
            dropout=gnn_dropout,
            connection_type=connection_type,
            k=gnn_k,
            threshold=gnn_threshold
        )

        # Store electrode info for reference
        self.electrode_names = self.spatial_gnn.electrode_names
        self.num_electrodes = self.spatial_gnn.num_electrodes

        # Activation and pooling
        self.elu = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, eegnet_pooling_1))

        # Block 3: Temporal separable convolution (unchanged)
        self.block3 = nn.Conv2d(F2, F2, (1, eegnet_separable_kernel_size), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avg_pool2 = nn.AvgPool2d((1, eegnet_pooling_2))

        # Optional ECA layer
        self.use_eca = use_eca
        if self.use_eca:
            self.eca = ECALayer(F2)

    def forward(self, x, debug_mode_flag=False):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, num_channels, time_steps)
        """
        # Block 1: Temporal convolution
        if debug_mode_flag:
            print('Shape of x before block1 of EEGNet-GNN: ', x.shape)

        x = self.block1(x)
        if debug_mode_flag:
            print('Shape of x after block1 of EEGNet-GNN: ', x.shape)

        x = self.bn1(x)

        # Block 2: Graph-based spatial processing
        if debug_mode_flag:
            print('Shape of x before spatial GNN: ', x.shape)

        x = self.spatial_gnn(x)
        if debug_mode_flag:
            print('Shape of x after spatial GNN: ', x.shape)

        x = self.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout(x)

        # Block 3: Temporal separable convolution
        if debug_mode_flag:
            print('Shape of x before block3 of EEGNet-GNN: ', x.shape)

        x = self.block3(x)
        if debug_mode_flag:
            print('Shape of x after block3 of EEGNet-GNN: ', x.shape)

        x = self.bn3(x)
        x = self.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout(x)

        if debug_mode_flag:
            print('Shape of x by the end of EEGNet-GNN: ', x.shape)

        # Apply ECA if enabled
        if self.use_eca:
            x = self.eca(x)
            if debug_mode_flag:
                print('Shape of x after ECA: ', x.shape)

        return x

class EEGTransformerNet(nn.Module):
    def __init__(self, nb_classes, sequence_length, eeg_chans=22,
                 F1=16, D=2, eegnet_kernel_size=32, dropout_eegnet=0.3, dropout_positional_encoding=0.3, eegnet_pooling_1=5, eegnet_pooling_2=5, 
                 MSA_num_heads = 8, flag_positional_encoding=True, transformer_dim_feedforward=2048, num_transformer_layers=6, use_eca=False):
        super(EEGTransformerNet, self).__init__()
        """
        F1 = the number of temporal filters
        F2 = number of spatial filters
        """

        F2 = F1 * D
        self.sequence_length_transformer = sequence_length//eegnet_pooling_1//eegnet_pooling_2

        self.eegnet = EEGNetGNN(eeg_chans=eeg_chans, F1=F1, eegnet_kernel_size=eegnet_kernel_size, D=D,
                             eegnet_pooling_1=eegnet_pooling_1, eegnet_pooling_2=eegnet_pooling_2, dropout=dropout_eegnet,use_eca=use_eca)
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
        if debug_mode_flag: print('Shape of x before EEGNet of EEGTransformerNet: ', x.shape)

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

        if debug_mode_flag: print('Shape of x after EEGNet of EEGTransformerNet: ', x.shape)
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
            use_eca=self.cfg.use_eca,
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
        # TODO this is not nice!
        x = x.float().to(self.device)
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
        # TODO this is not nice!
        x = x.float().to(self.device)
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
        # TODO this is not nice!
        x = x.float().to(self.device)
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