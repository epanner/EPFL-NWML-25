from wandb import Api

api = Api()
# Fetch that specific run
run = api.run("veit-epfl-epfl/eeg-gnn/0fi66hg5") # TODO configure run name

# Download exactly that file
f = run.file("checkpoints/best-checkpoint-2025-05-15_15-42-19.ckpt") # TODO configure checkpoint
local_path = f.download(replace=True)

print(f"Downloaded {f.name} → {local_path}")