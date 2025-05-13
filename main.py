from datetime import datetime
from pathlib import Path
import hydra
from hydra.utils import instantiate
import pandas as pd
import torch
# from model.gnn import EEGGNN
from model.eeg_transformer import EEGGNN
from preprocessing.preprocessing import PREPROCESSING_REGISTRY
from functools import reduce
from omegaconf import DictConfig
from seiz_eeg.dataset import EEGDataset
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

def compose_transforms(transforms):
    def composed(x):
        return reduce(lambda data, f: f(data), transforms, x)
    return composed

def instantiate_preprocessing(cfg_step):
    cls = PREPROCESSING_REGISTRY[cfg_step["_target_"]]
    cfg_step = {k: v for k, v in cfg_step.items() if k != "_target_"}
    return cls(**cfg_step)


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.train.seed)

    transforms = [instantiate_preprocessing(step) for step in cfg.preprocessing.steps]
    transform_fn = compose_transforms(transforms)
    
    DATA_ROOT = Path(cfg.dataset.data_path)
    clips_tr = pd.read_parquet(DATA_ROOT / cfg.dataset.train_set / "segments.parquet")

    dataset_tr = EEGDataset(
        clips_tr,
        signals_root=DATA_ROOT / cfg.dataset.train_set,
        signal_transform=transform_fn,
        prefetch=cfg.train.prefetch_dataset
    )
    wandb.init(project="eeg-gnn", config=dict(cfg))
    wandb_logger = WandbLogger(project="nml-project", config=dict(cfg))


    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"best-checkpoint-{timestamp}",
        verbose=True,
        monitor=cfg.checkpoint.monitor,
        save_top_k=cfg.checkpoint.save_top_k,
        mode="min"
    )
    early_stop_cb = pl.callbacks.EarlyStopping(monitor=cfg.early_stopping.monitor,
                                               patience=cfg.early_stopping.patience)

    
    loader_tr = DataLoader(dataset_tr, batch_size=cfg.train.batch_size, shuffle=True)
    model = EEGGNN(cfg.model)

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, early_stop_cb],
        gradient_clip_val=cfg.train.gradient_clip_val,
    )

    trainer.fit(model, loader_tr)
    wandb.save(checkpoint_callback.best_model_path) #TODO auto logging

    clips_te = pd.read_parquet(DATA_ROOT / cfg.dataset.test_set / "segments.parquet")

    dataset_te = EEGDataset(
        clips_te,  # Your test clips variable
        signals_root=DATA_ROOT
        / cfg.dataset.test_set,  # Update this path if your test signals are stored elsewhere
        signal_transform=transform_fn,  # You can change or remove the signal_transform as needed
        prefetch=cfg.train.prefetch_dataset,  # Set to False if prefetching causes memory issues on your compute environment
        return_id=True,  # Return the id of each sample instead of the label
    )
    loader_te = DataLoader(dataset_te, batch_size=cfg.train.batch_size, shuffle=True)

    model.eval()
    all_predictions = []
    all_ids = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for batch in loader_te:
            x_batch, x_ids = batch
            x_batch = x_batch.float().to(device)
            logits = model(x_batch)
            predictions = (logits > 0).int().cpu().numpy()
            all_predictions.extend(predictions.flatten().tolist())
            all_ids.extend(list(x_ids))


    wandb.finish()

if __name__ == '__main__':
    main()
