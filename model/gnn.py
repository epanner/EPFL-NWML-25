import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import lightning as pl

class EEGGNN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(19, cfg.model.hidden_channels))
        for _ in range(cfg.model.num_layers - 1):
            self.convs.append(GCNConv(cfg.model.hidden_channels, cfg.model.hidden_channels))
        self.fc = torch.nn.Linear(cfg.model.hidden_channels, 1)
        self.dropout = cfg.model.dropout
        self.lr = cfg.learning_rate
        self.optimizer_name = cfg.optimizer.name

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x.mean(dim=0))  # Pooling (mean)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x.x, x.edge_index)
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(), y.float())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.optimizer_name) #raises a error!
        return optimizer_class(self.parameters(), lr=self.lr)
    
    # Spatial then temporal
    # GNN(GAN, GCNConv, GraphTransformer, Graphspectral, FlatVector) -> (LSTM/GRU, BI-LSTM/GRU, Transformer)

    # Was muss noch gemacht werden?
    # - Graph einlesen
    # - Configuration ändern -> klomplette Klasse für eine Modelkonfiguration!(Veit)
    # - Metrics zur Klasse hinzufügen(Veit)
    # - Chatgpt fragen welche Elememte
    #   - beste zuerst implementieren
    #   - Transformer -> 
    #   - BI-LSTM -> 
    #