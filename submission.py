
import os
import hydra
import torch
from seiz_eeg.dataset import EEGDataset
import pandas as pd
from pathlib import Path
from datetime import datetime
from preprocessing.preprocessing import create_signal_transformer
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from omegaconf import DictConfig
import lightning.pytorch as pl
from model.eeg_transformer import EEGGNN
from tqdm import tqdm

def generate_submission(cfg, model, transform_fn=None, fix_underscores=False):
    DATA_ROOT = Path(cfg.dataset.data_path)
    clips_te = pd.read_parquet(DATA_ROOT / cfg.dataset.test_set / "segments.parquet")

    if transform_fn is None:
        transform_fn = create_signal_transformer(cfg.preprocessing.steps)

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
        # for batch in enumerate(tqdm(loader_te)):
            x_batch, x_ids = batch
            x_batch = x_batch.float().to(device)
            logits = model(x_batch)
            # print(logits)
            predictions = model.prediction(logits)
            all_predictions.extend(predictions.flatten().tolist())
            all_ids.extend(list(x_ids))
            # print(f"    → total IDs so far={len(all_ids)}, total preds so far={len(all_predictions)}")

    print(f"Final count — IDs: {len(all_ids)}, Predictions: {len(all_predictions)}")
    # Create a DataFrame for Kaggle submission with the required format: "id,label"
    # WORKAROUND problem is dataloader sets between each char in id string a _
    if fix_underscores:
        all_ids = list(map(lambda s: s[::2], all_ids))
    submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save the DataFrame to a CSV file without an index
    try:
        os.mkdir("submission")
    except Exception as e:
        print(f"An error occurred: {e}")    
    submission_df.to_csv(f"submission/submission_seed_{now}.csv", index=False)
    OmegaConf.save(cfg, f"submission/config_{now}.yaml")

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.train.seed)
    print(cfg.checkpoint_path)
    model = EEGGNN.load_from_checkpoint(cfg.checkpoint_path)
    generate_submission(cfg, model, fix_underscores=True)

# run like
# python submission.py \
# +checkpoint_path=/home/veit/Uni/Lausanne/NML/EPFL-NWML-25/checkpoints/best-checkpoint-2025-05-14_18-56-10.ckpt


# If you want to use different config
# --config-path /new/path/to/conf \


if __name__ == '__main__':
    main()
