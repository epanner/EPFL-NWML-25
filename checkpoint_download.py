from wandb import Api

api = Api()
# Fetch that specific run
run = api.run("veit-epfl-epfl/eeg-gnn/ijn0fj17") # TODO configure run name

# Download exactly that file
f = run.file("checkpoints/best-checkpoint-2025-05-14_18-56-10.ckpt") # TODO configure checkpoint
local_path = f.download(replace=True)

print(f"Downloaded {f.name} → {local_path}")