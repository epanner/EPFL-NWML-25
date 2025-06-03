from wandb import Api

api = Api()
# Fetch that specific run
run = api.run("veit-epfl-epfl/eeg-gnn/lpnbwfwu") # TODO configure run name

print(run.name)
# Download exactly that file
f = run.file("checkpoints/best-checkpoint-EEGGNN_Binary-2025-06-03_13-50-20.ckpt") # TODO configure checkpoint
local_path = f.download(replace=True)

print(f"Downloaded {f.name} → {local_path}")