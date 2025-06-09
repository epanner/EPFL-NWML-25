from datetime import datetime
from pathlib import Path
import hydra
import pandas as pd
import torch
# from model.gnn import EEGGNN
from preprocessing.preprocessing import create_signal_transformer
from omegaconf import DictConfig
from seiz_eeg.dataset import EEGDataset
from sklearn.model_selection import train_test_split
import wandb
import lightning.pytorch as pl  
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from submission import generate_submission
from model.model import MODEL_REGISTRY

def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: dict = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.train.seed)

    transform_fn = create_signal_transformer(cfg.preprocessing.steps)
    
    DATA_ROOT = Path(cfg.dataset.data_path)
    clips_tr = pd.read_parquet(DATA_ROOT / cfg.dataset.train_set / "segments.parquet")

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
    run_name = f"test_gnn"
    # run_name = f"kernel_size_{cfg.model.eegnet_kernel_size}_f1_{cfg.model.F1}"                                                                                                                                                              
    wandb.init(project="eeg-gnn",
               config=flatten_dict(cfg),
               name=run_name, 
               reinit=True)
    wandb_logger = WandbLogger(project="nml-project", config=dict(cfg))

    
    loader_tr = DataLoader(dataset_tr, batch_size=cfg.train.batch_size, shuffle=True)#, num_workers=11)
    if not cfg.train.comp_mode:
        loader_vl = DataLoader(dataset_vl, batch_size=cfg.train.batch_size, shuffle=False)#, num_workers=11)

    # Quick-and-dirty sanity-check
    batch = next(iter(loader_tr))
    x = batch[0]
    print("min / max :", x.min().item(), x.max().item())
    print("any NaN?  :", torch.isnan(x).any().item())
    print("any Inf?  :", torch.isinf(x).any().item())

    # model = EEGGNN(cfg.model)
    model = MODEL_REGISTRY[cfg.model["_target_"]](cfg.model)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"best-checkpoint-{model}-{timestamp}",
        verbose=True,
        monitor=cfg.checkpoint.monitor_comp_mode if cfg.train.comp_mode else cfg.checkpoint.monitor,
        save_top_k=cfg.checkpoint.save_top_k,
        mode="min"
    )
    early_stop_cb = pl.callbacks.EarlyStopping(monitor=cfg.early_stopping.monitor_comp_mode if cfg.train.comp_mode else cfg.early_stopping.monitor,
                                               patience=cfg.early_stopping.patience)

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, early_stop_cb],
        gradient_clip_val=cfg.train.gradient_clip_val,
        num_sanity_val_steps=2,
    )

    trainer.fit(model, loader_tr, loader_vl if not cfg.train.comp_mode else None)
    wandb.save(checkpoint_callback.best_model_path) #TODO auto logging

    generate_submission(cfg, model, transform_fn=transform_fn)

    wandb.finish()

if __name__ == '__main__':
    main()
