from datetime import datetime
from pathlib import Path
import hydra
import pandas as pd
import torch
# from model.gnn import EEGGNN
from model.eeg_transformer import EEGGNN
from preprocessing.preprocessing import create_signal_transformer
from omegaconf import DictConfig
from seiz_eeg.dataset import EEGDataset
from sklearn.model_selection import train_test_split
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from submission import generate_submission

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.train.seed)

    transform_fn = create_signal_transformer(cfg.preprocessing.steps)
    
    DATA_ROOT = Path(cfg.dataset.data_path)
    clips_tr = pd.read_parquet(DATA_ROOT / cfg.dataset.train_set / "segments.parquet")

    # TODO Wie weit muss man ins Fenster davor und danach schauen gerade auch fürs Kurven glätten relevant oder?
    # Oder schaut keiner in mehrere Fenster?!
    if not cfg.train.comp_mode:
        clips_tr, clips_val = train_test_split(
            clips_tr,
            test_size=0.2,        # 20% for validation
            random_state=cfg.train.seed,
            shuffle=True,
            stratify=clips_tr['label']  # if you have a label column
        )

        dataset_vl = EEGDataset(
            clips_val,
            signals_root=DATA_ROOT / cfg.dataset.train_set,
            signal_transform=transform_fn,
            prefetch=cfg.train.prefetch_dataset
        )
   
    dataset_tr = EEGDataset(
        clips_tr,
        signals_root=DATA_ROOT / cfg.dataset.train_set,
        signal_transform=transform_fn,
        prefetch=cfg.train.prefetch_dataset
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")                                                                                                                              
                                                                                                                                                                                  
    wandb.init(project="eeg-gnn", config=dict(cfg))
    wandb_logger = WandbLogger(project="nml-project", config=dict(cfg))
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"best-checkpoint-{timestamp}",
        verbose=True,
        monitor=cfg.checkpoint.monitor_comp_mode if cfg.train.comp_mode else cfg.checkpoint.monitor,
        save_top_k=cfg.checkpoint.save_top_k,
        mode="min"
    )
    early_stop_cb = pl.callbacks.EarlyStopping(monitor=cfg.early_stopping.monitor_comp_mode if cfg.train.comp_mode else cfg.early_stopping.monitor,
                                               patience=cfg.early_stopping.patience)

    
    loader_tr = DataLoader(dataset_tr, batch_size=cfg.train.batch_size, shuffle=True)#, num_workers=11)
    if not cfg.train.comp_mode:
        loader_vl = DataLoader(dataset_vl, batch_size=cfg.train.batch_size, shuffle=False)#, num_workers=11)

    model = EEGGNN(cfg.model)

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, early_stop_cb],
        gradient_clip_val=cfg.train.gradient_clip_val,
        num_sanity_val_steps=2
    )

    trainer.fit(model, loader_tr, loader_vl if not cfg.train.comp_mode else None)
    wandb.save(checkpoint_callback.best_model_path) #TODO auto logging

    generate_submission(cfg, model, transform_fn=transform_fn)

    wandb.finish()

if __name__ == '__main__':
    main()
