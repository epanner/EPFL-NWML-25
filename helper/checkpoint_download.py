from wandb import Api

api = Api()
# Fetch that specific run
run = api.run("veit-epfl-epfl/eeg-gnn/qgsvzg8s") # TODO configure run name

print(run.name)
# Download exactly that file
f = run.file("checkpoints/best-checkpoint-NeuroGNN-2025-06-09_16-40-08.ckpt") # TODO configure checkpoint
local_path = f.download(replace=True)

print(f"Downloaded {f.name} → {local_path}")