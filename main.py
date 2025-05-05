import datetime
from pathlib import Path
import hydra
from hydra.utils import instantiate
import pandas as pd
from model.gnn import EEGGNN
from preprocessing import PREPROCESSING_REGISTRY
from functools import reduce
from omegaconf import DictConfig
from seiz_eeg.dataset import EEGDataset
import wandb
import lightning as pl
from torch.utils.data import DataLoader

def compose_transforms(transforms):
    def composed(x):
        return reduce(lambda data, f: f(data), transforms, x)
    return composed

def instantiate_preprocessing(cfg_step):
    cls = PREPROCESSING_REGISTRY[cfg_step["_target_"]]
    cfg_step = {k: v for k, v in cfg_step.items() if k != "_target_"}
    return cls(**cfg_step)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    transforms = [instantiate_preprocessing(step) for step in cfg.preprocessing]
    transform_fn = compose_transforms(transforms)
    
    # TODO this could also all added to the config
    data_path = "/content/networkML"
    DATA_ROOT = Path(data_path)
    clips_tr = pd.read_parquet(DATA_ROOT / "train/train/segments.parquet")

    dataset_tr = EEGDataset(
        clips_tr,
        signals_root=DATA_ROOT / "train",
        signal_transform=transform_fn,
        prefetch=cfg.prefetch_dataset
    )
    
    wandb.init(project="eeg-gnn", config=dict(cfg))
    wandb_logger = pl.loggers.WandbLogger(project="nml-project", config=dict(cfg))


    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"best-checkpoint-{timestamp}",
        save_top_k=1,
        verbose=True,
        monitor="train_loss",
        mode="min"
    )
    
    loader_tr = DataLoader(dataset_tr, batch_size=cfg.batch_size, shuffle=True)
    model = EEGGNN(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.num_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, loader_tr)
    wandb.save(checkpoint_callback.best_model_path)

    clips_te = pd.read_parquet(DATA_ROOT / "test/test/segments.parquet")

    dataset_te = EEGDataset(
        clips_te,  # Your test clips variable
        signals_root=DATA_ROOT
        / "test",  # Update this path if your test signals are stored elsewhere
        signal_transform=transform_fn,  # You can change or remove the signal_transform as needed
        prefetch=cfg.prefetch_dataset,  # Set to False if prefetching causes memory issues on your compute environment
        return_id=True,  # Return the id of each sample instead of the label
    )
    loader_te = DataLoader(dataset_te, batch_size=cfg.batch_size, shuffle=True)

    model.eval()
    all_predictions = []
    all_ids = []

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)

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
