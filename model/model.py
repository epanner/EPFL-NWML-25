from model.eeg_transformer import EEGTranformer_Binary
from model.eeg_gat import EEG_GAT
from model.gnn import GCN_LSTM
from model.neuro_gnn import NeuroGNN

MODEL_REGISTRY = {
    "EEGTransformer": EEGTranformer_Binary,
    "EEGGraph": EEG_GAT,
    "NeuroGNN": NeuroGNN,
    "GCNLSTM": GCN_LSTM,
}
