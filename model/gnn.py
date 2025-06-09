from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch as Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool         # spatial :contentReference[oaicite:4]{index=4}
import lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

from model.eeg_transformer import EEGTranformer
from model.neuro_constants import INCLUDED_CHANNELS  # temporal metrics :contentReference[oaicite:5]{index=5}
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import dense_to_sparse

class GCNEncoder(nn.Module):
    """Stacked GCN → node embeddings."""
    def __init__(self, in_channels: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        convs = [GCNConv(in_channels, hidden)]
        convs += [GCNConv(hidden, hidden) for _ in range(num_layers - 1)]
        self.convs = nn.ModuleList(convs)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, self.dropout, self.training)
        return x


# class GraphLSTMNet(pl.LightningModule):
#     """
#     Spatial (GCN) + Temporal (Bi-LSTM) network for 12-second EEG clips.
#     Input  shape : (B, T, N, F)
#     Output shape : (B,) logits  (BCEWithLogitsLoss inside)
#     """
#     def __init__(self, cfg):
#         super().__init__()
#         self.save_hyperparameters(cfg)

#         H = cfg.model.hidden_channels
#         self.gcn = GCNEncoder(
#             in_channels=cfg.model.in_channels,   #  e.g. 1 after FFT-log amp
#             hidden=H,
#             num_layers=cfg.model.num_layers,
#             dropout=cfg.model.dropout)

#         # Temporal block
#         self.lstm = nn.LSTM(
#             input_size=H,
#             hidden_size=H,
#             num_layers=2,
#             bidirectional=True,
#             batch_first=True,
#             dropout=cfg.model.dropout)

#         self.classifier = nn.Linear(H * 2, 1)    # 2× for Bi-directional
#         self.criterion = nn.BCEWithLogitsLoss()

#         # optimiser settings
#         self.lr = cfg.optim.lr
#         self.optimizer_name = cfg.optim.name

#     # ------------------------------------------------------------------ #
#     # Forward pass
#     # ------------------------------------------------------------------ #
#     def forward(self, x, edge_index, batch_vec):
#         """
#         x          : (B, T, N, F)
#         edge_index : (2, E) static montage
#         batch_vec  : (B·T·N,) maps each node to its (graph) id
#         """
#         B, T, N, F = x.shape
#         # Flatten B & T ⇒ graphs live in rows
#         x = x.reshape(B * T, N, F).reshape(-1, F)          # node matrix
#         h = self.gcn(x, edge_index)                        # (B·T·N, H)

#         # global_mean_pool needs node ↦ graph map
#         h_graph = global_mean_pool(h, batch_vec)           # (B·T, H)

#         # restore temporal order and batch
#         h_graph = h_graph.reshape(B, T, -1)                # (B, T, H)
#         lstm_out, _ = self.lstm(h_graph)                   # (B, T, 2H)

#         last = lstm_out[:, -1]                             # (B, 2H)
#         return self.classifier(last).squeeze(1)            # (B,)

#     # ------------------------------------------------------------------ #
#     # Lightning boiler-plate
#     # ------------------------------------------------------------------ #
#     def common_step(self, batch, stage: str):
#         data, y = batch                                   # adapt to your DataLoader
#         logits = self(data.x, data.edge_index, data.batch)
#         loss   = self.criterion(logits, y.float())

#         if stage == "train":
#             self.train_acc.update(torch.sigmoid(logits), y.int())
#             self.log("train_loss", loss, prog_bar=True)
#             self.log("train_acc",  self.train_acc, prog_bar=True)
#         else:
#             self.val_acc.update(torch.sigmoid(logits), y.int())
#             self.val_f1.update(torch.sigmoid(logits), y.int())
#             self.log(f"{stage}_loss", loss, prog_bar=True)
#         return loss

#     def training_step(self, batch, batch_idx):
#         return self.common_step(batch, "train")

#     def validation_step(self, batch, batch_idx):
#         self.common_step(batch, "val")

#     def on_validation_epoch_end(self):
#         self.log("val_acc", self.val_acc.compute(), prog_bar=True)
#         self.log("val_f1",  self.val_f1.compute(),  prog_bar=True)
#         self.val_acc.reset(), self.val_f1.reset()

#     # ------------------------------------------------------------------ #
#     # Optimiser & LR scheduler
#     # ------------------------------------------------------------------ #
#     def configure_optimizers(self):
#         try:
#             OptimCls = getattr(torch.optim, self.optimizer_name)   # e.g. "Adam" :contentReference[oaicite:6]{index=6}
#         except AttributeError as e:
#             raise ValueError(f"Unknown optimiser: {self.optimizer_name}") from e

#         optimizer = OptimCls(self.parameters(), lr=self.lr)
#         return optimizer

def get_adjacency_matrix(distance_df, sensor_ids, dist_k=0.9):
    """
    Args:
        distance_df: data frame with three columns: [from, to, distance].
        sensor_ids: list of sensor ids.
        dist_k: threshold for graph sparsity
    Returns:
        adj_mx: adj
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.    
    adj_mx = np.exp(-np.square(dist_mx / std))
    adj_mx[dist_mx > dist_k] = 0
   
    return adj_mx, sensor_id_to_ind, dist_mx

class GCN_LSTM(EEGTranformer):
    def __init__(self, cfg, ):
        super().__init__(cfg, create_model=False)
        self.save_hyperparameters()

        H = cfg.hidden_channels
        self.gcn = GCNEncoder(
            in_channels=cfg.in_channels,
            hidden=H,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout)

        # Temporal block
        self.lstm = nn.LSTM(
            input_size=H,
            hidden_size=H,
            num_layers=cfg.lstm_layers,
            bidirectional=cfg.bidirectional,
            batch_first=True,
            dropout=cfg.dropout)

        class_layer = 2*H if cfg.bidirectional else H
        self.classifier = nn.Linear(class_layer, 1)    # 2× for Bi-directional
        self.criterion = nn.BCEWithLogitsLoss()

        # optimiser settings
        self.lr = cfg.optimizer.lr
        self.optimizer_name = cfg.optimizer.name

        # create adjacency matrix
        csv_path = Path(cfg.distance_csv_root) / cfg.distance_csv_path
        dist_df = pd.read_csv(csv_path)
        adj_mat,_ ,_ = get_adjacency_matrix(dist_df, INCLUDED_CHANNELS)


        edge_index, _   = dense_to_sparse(torch.tensor(adj_mat, dtype=torch.float32))
        self.register_buffer("edge_index_base", edge_index)   # moves to GPU
        self.num_nodes       = adj_mat.shape[0]               # = 19
        self.resample_fs     = cfg.resample_fs                # e.g. 250 Hz
        self.sequence_sec    = cfg.sequence_length            # e.g. 12 s

        # self.edge_index, self.edge_attr = pyg_utils.dense_to_sparse(torch.tensor(self.adj_mx))


    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor (B, L, 19)
            L should equal sequence_sec * resample_fs (or will be trimmed).
        """
        B, L, N = x.shape
        assert N == self.num_nodes, f"expected {self.num_nodes} EEG channels"

        F = self.resample_fs                       # samples per 1-s window
        T = L // F                                 # number of windows (12)
        x = x[:, :T*F, :].view(B, T, F, N)         # (B, T, F, N)
        x = x.permute(0, 1, 3, 2).contiguous()     # (B, T, N, F)

        # ---------- flatten graphs for PyG ---------- #
        x_nodes = x.reshape(B * T * N, F)          # (B·T·N, F)

        EI = self.edge_index_base                  # (2, E)
        E  = EI.size(1)
        offsets  = torch.arange(B*T, device=x.device) * N
        edge_index = EI.repeat(1, B*T) + offsets.repeat_interleave(E)

        batch_vec = torch.arange(B*T, device=x.device).repeat_interleave(N)

        # ---------- GCN -> pooled graph emb ---------- #
        h = self.gcn(x_nodes, edge_index)          # (B·T·N, H)
        g = global_mean_pool(h, batch_vec)         # (B·T, H)

        # ---------- LSTM over time ------------------- #
        g = g.view(B, T, -1)                       # (B, T, H)
        lstm_out, _ = self.lstm(g)                 # (B, T, 2H)
        logits = self.classifier(lstm_out[:, -1])  # last step  (B, 1)
        return logits.squeeze(1)                   # (B,)
        
    def prediction(self, logits: Tensor):
        # 1) get probabilities
        probs = torch.sigmoid(logits)
        # 2) make binary predictions
        preds = (probs >= 0.5).long()
        return preds
    
    def loss_func(self, x: Tensor, y:Tensor):
        return self.criterion(x, y.float())
    
    def configure_optimizers(self):
        # TODO decay is not set here
        try:
            OptimCls = getattr(torch.optim, self.optimizer_name)
        except AttributeError as e:
            raise ValueError(f"Unknown optimiser: {self.optimizer_name}") from e

        optimizer = OptimCls(self.parameters(), lr=self.lr)
        return optimizer