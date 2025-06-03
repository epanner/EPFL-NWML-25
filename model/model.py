from model.eeg_transformer import EEGGNN, EEGGNN_Binary
from model.eeg_gat import EEG_GAT
MODEL_REGISTRY = {
    "EEGTransformer": EEGGNN_Binary,
    "EEGGraph": EEG_GAT,
}
