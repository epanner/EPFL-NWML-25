from pathlib import Path
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from torch_geometric.nn import GATConv
import pandas as pd
from model.eeg_transformer import EEGTranformer_Binary
debug_mode_flag = False


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
    def load_distance_csv(csv_path: Path):
        """
        Load electrode distances from CSV file

        Args:
            csv_path: Path to CSV file with columns ['from', 'to', 'distance']

        Returns:
            electrode_names: List of electrode names
            distance_matrix: Symmetric distance matrix
        """

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

    def __init__(self, in_channels, out_channels, distance_csv_path: Path =None,
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
                 eegnet_dropout=0.5, use_eca=False, gnn_heads=4, gnn_dropout=0.1,
                 connection_type='knn', gnn_k=4, gnn_threshold=0.8):
        super(EEGNetGNN, self).__init__()

        F2 = F1 * D
        self.dropout = nn.Dropout(eegnet_dropout)

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

        #TODO Is here a norm missing?!

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


class EEG_GAT(EEGTranformer_Binary):
    def __init__(self, cfg):
        super().__init__(cfg, create_model=False)
        self.save_hyperparameters()
    
        self.model = EEGNetGNN(
            eeg_chans=cfg.num_eeg_channels,
            F1=cfg.F1,
            D=cfg.D,
            eegnet_kernel_size=cfg.eegnet_kernel_size,
            eegnet_pooling_1=cfg.eegnet_pooling_1,
            eegnet_pooling_2=cfg.eegnet_pooling_2,
            dropout_eegnet=cfg.dropout_eegnet,

            distance_csv_path=Path(cfg.distance_csv_root) / cfg.distance_csv_path,
            use_eca=cfg.use_eca,
            gnn_heads=cfg.gnn_heads,
            gnn_dropout=cfg.gnn_dropout,
            gnn_threshold=cfg.gnn_threshold,
            connection_type=cfg.connection_type,
        )

    def forward(self, x):
        logits = self.model(x)
        return logits.squeeze(-1)