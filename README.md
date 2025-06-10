# EEG Seizure Detection (EPFL-NWML-25)

A deep learning framework for automatic epileptic seizure detection in EEG recordings.

## Installation

Ensure you have the following key dependencies installed:

* **Hydra** – for hierarchical configuration management and command-line overrides.
* **PyTorch Lightning** – for organizing training loops and experiment management.
* **PyTorch Geometric** – for graph neural network layers (used in GCN-based models).

All required packages (including the above) are listed in `requirements.txt`. You can install them with pip:

```bash
pip install -r requirements.txt
```

*(It's recommended to use a Python 3.x environment with PyTorch compatible with your CUDA version for GPU support.)*

To enable automatic logging with Weights & Biases, first log in:
```bash
wandb login
```

## Dataset

This project uses the **Temple University Hospital EEG Seizure Corpus (TUSZ)** as the primary dataset. TUSZ is one of the largest publicly available EEG datasets for seizure detection, containing thousands of hours of continuous scalp EEG recordings from epilepsy patients. Each recording is annotated with seizure events by experts, providing a rich resource for training and evaluating detection algorithms. The data encompasses a wide variety of seizure types and patient conditions, making it a realistic and challenging benchmark for machine learning models in an academic and clinical context.

**Note:** The Temple EEG dataset is not packaged with this repository. You should download it from the official source and set the correct path in the configuration (or via command-line override). The dataset configuration is defined in the Hydra config file `config/dataset/temple_dataset.yaml`.W An excerpt from this config file is shown below:

```yaml
# config/dataset/temple_dataset.yaml
  _target_: EEGDataset
  data_path: "/home/veit/Uni/Lausanne/NML/EPFL-NWML-25/content/networkML" 
  train_set: "train/train" #train data dir
  test_set: "test/test"    #test data dir
  distance_csv_root: "/home/veit/Uni/Lausanne/NML/EPFL-NWML-25"
  distances_set: "distances_3d.csv"
  train_test_split: 0.2
```

You can modify this YAML or override its values at runtime to point to your local data. For instance, to use a custom data directory without editing the file, you can pass `dataset.data_dir=/your/data/path` as a command-line argument (see below).

## Running Training

The training process is managed by the **`main.py`** script, which utilizes Hydra for configuration. You can launch training by specifying a particular configuration file (which sets the model, dataset, and training parameters) and optional overrides. For example, to run training with the default settings for a GCN+BiLSTM model:

```bash
python main.py --config-name gcn_run_lstm.yaml
```

This will load `config/gcn_run_lstm.yaml` along with its default config groups (which include using the Temple dataset and specific model settings). Hydra allows you to override configuration options directly from the command line. For instance, if you need to change the dataset path or any parameter without editing the config files, you can append overrides:

```bash
python main.py \
dataset.data_path=$your_dataset_path$ \
dataset.train_set=train \
dataset.test_set=test \
dataset.distance_csv_root=$your_distance_path$ \
--config-name=gcn_run_lstm.yaml
```

In the above example, the dataset directory and the number of training epochs are overridden. **Hydra** will compose the final configuration from the YAML files and your overrides, then start the PyTorch Lightning training run accordingly.

## Models and Training Pipeline

The repository implements three neural network models for seizure detection. Each model architecture is briefly described below. During training, PyTorch Lightning’s `Trainer` handles the training loop, validation, checkpointing, and logging. All models take EEG time-series data as input and output seizure predictions, but they differ in how they learn spatial and temporal features:

* **GCN + Bi-LSTM**: This model combines a Graph Convolutional Network with a Bidirectional LSTM. EEG electrode channels are treated as nodes in a graph (with edges defined by physical or functional adjacency), and graph convolution layers learn spatial relationships between channels. The output of the GCN is then fed into a Bi-LSTM, which captures temporal dynamics in the EEG signal (forward and backward in time). This spatio-temporal modeling allows the network to learn where (which channels) and when seizures occur.
  *Training command:*

  ```bash
  python main.py --config-name gcn_run_lstm.yaml
  ```

* **Transformer**: This model employs a transformer architecture with self-attention mechanisms to model EEG sequences. The Transformer processes the multichannel EEG data, capturing long-range temporal dependencies and interactions between channels via multi-head attention layers. It treats the EEG signal as a sequential data structure and learns complex patterns of activity indicative of seizures, without needing an explicit recurrent module. This approach can capture global context across the entire recording segment.
  *Training command:*

  ```bash
  python main.py --config-name eeg_transformer_run.yaml
  ```

* **NeuroGNN**: The NeuroGNN is a specialized Graph Neural Network tailored for EEG data. It integrates neurological domain knowledge (such as the physical layout or known connectivity of EEG electrodes) into its graph structure. This model uses advanced GNN layers (e.g., graph convolutions or attentional edges) to learn spatial features, potentially combined with temporal processing internally. The architecture is designed to exploit the brain’s network characteristics, aiming to improve detection performance and interpretability for seizure activity.
  *Training command:*

  ```bash
  python main.py --config-name eeg_gnn_run.yaml
  ```

Model checkpoints and logs will be saved automatically (by PyTorch Lightning) to the output directory (see **Outputs** section below).

## Hyperparameter Sweeps & Tuning

For extensive experimentation, you can perform hyperparameter sweeps using Hydra’s **multi-run** capability or by editing the config files:

* **Hydra Multi-run**: To run multiple configurations in one go, use the `--multirun` flag with comma-separated override values. Hydra will launch a series of training runs for each combination of the specified values. For example:

  ```bash
  python main.py --multirun training.max_epochs=50,100 model.learning_rate=0.001,0.0005 --config-name eeg_gnn_run.yaml
  ```

  This command will run four experiments covering all combinations of `max_epochs` {50, 100} and learning rate {0.001, 0.0005}. Each run’s outputs will be saved in a separate folder (Hydra will typically create a directory like `multirun/` or unique timestamped folders for each run).
* **Tuning via Config**: Hydra also supports more advanced sweep configurations (for example, see `config/sweep_eeg.yaml` in the repository for a preset sweep setup). You can run a sweep config by specifying it as the config name with `--multirun`. Always review the output logs to monitor each run’s performance and identify the best hyperparameter settings.
 ```bash
  python main.py -config-name=sweep_eeg.yaml --multirun
  ```

## Project Structure

The repository is organized into several modules to keep components modular and maintainable:

* **`config/`** – Configuration files for Hydra. This folder contains:

  * **`dataset/`**: dataset configs (e.g. `temple_dataset.yaml` defines dataset paths and parameters).
  * **`model/`**: model architecture and hyperparameter configs (e.g. `gcn_conv.yaml`, `eeg_transformer.yaml`, etc.).
  * **`preprocessing/`**: preprocessing method configs (e.g. `fft_filter.yaml` for filtering settings).
  * **(root of config)**: High-level experiment configs tying everything together, such as `gcn_run_lstm.yaml`, `eeg_transformer_run.yaml`, `eeg_gnn_run.yaml` for each model’s training run, and `sweep_eeg.yaml` for multi-run sweeps.
* **`model/`** – Implementation of model architectures and a model registry:

  * `model.py` – Model selection logic or registry. This file likely contains a factory method or mappings to instantiate the correct model class based on the config.
  * `eeg_transformer.py` – Defines the Transformer-based model class.
  * `neuro_gnn.py` – Defines the NeuroGNN model class.
  * `gnn.py` / `eeg_gat.py` – Contains graph neural network layers or utilities (e.g. Graph Convolution or Graph Attention implementations used by the models).
  * `neuro_constants.py` – Contains constants or metadata (possibly electrode positions, etc., used by NeuroGNN).
* **`preprocessing/`** – Preprocessing pipeline components for EEG data:

  * `preprocessing.py` – Base `Preprocessing` class and utilities for data transformations (normalization, filtering, etc.).
  * `preprocessing_fft.py`, `preprocessing_utils.py`, etc. – Specific preprocessing implementations (e.g., FFT-based filters, artifact removal) that inherit from the base class.
  * These modules are used to prepare raw EEG signals before feeding them into models (as configured in the YAML files).
* **`helper/`** – Utility scripts and helpers:

  * `submission.py` – Logic for generating the submission file from model predictions (run after training/evaluation).
  * `checkpoint_download.py` – (If present) a utility to download or manage model checkpoints (e.g., if pre-trained weights or best model need to be handled specially).
* **Root files**:

  * `main.py` – **Entry point** of the project. This script parses the Hydra config, initializes the dataset and model, and kicks off training (using PyTorch Lightning’s `Trainer`). It likely also triggers validation/testing and submission file creation.
  * `requirements.txt` – List of Python dependencies for the project.
  * `README.md` – Project documentation (you are reading an improved version of it).
  * `distances_3d.csv` (if present) – Possibly contains EEG electrode coordinate data, used for building graph connections between channels in GCN/GNN models.
  * `outputs/` and `multirun_logs/` – These directories are created when you run training. Hydra/Lightning will save logs, model checkpoints, and outputs here (each run gets a unique subfolder by timestamp or an index).

**Adding New Models or Preprocessing Modules:** If you plan to extend the repository with your own model architecture or preprocessing technique, see the guidance below.

## Extending the Project

### Adding a New Model

You can add custom models to the framework by following these steps:

1. **Implement the Model Class:** Create a new Python file in the `model/` directory (e.g., `my_model.py`) and implement your model. Your class should either subclass **PyTorch Lightning’s `LightningModule`** or inherit from an existing base class like `EEGTransformer` (if your model’s training step logic can reuse or extend it). Make sure to define the `forward` method and any required training/validation logic.
2. **Register the Model:** Open `model.py` and register your model so the pipeline can instantiate it from the config. This involve adding an entry in a dictionary to return your model when selected. For example, if configs use a name key (like `model.name: "MyModel"`), ensure `model.py` knows how to create an instance of `MyModel` when that name is specified.
3. **Create a Config:** Add a YAML config for your model under `config/model/`. This file should contain hyperparameters specific to your model (layer sizes, dropout rates, etc.). You can look at existing configs (like `eeg_transformer.yaml`) as a template. Also, you might create a new run config YAML (similar to `gcn_run_lstm.yaml`) that includes your model and any dataset/training settings unique to it.
4. **Update Dependencies:** If your model requires additional dependencies, update `requirements.txt` accordingly and mention them in the README if needed.

Once these steps are done, you can run training with your model by specifying the new config (e.g., `python main.py --config-name my_model_run.yaml`).

### Adding a New Preprocessing Step

The pipeline allows custom preprocessing of EEG signals (e.g., filtering, feature extraction) before feeding data to models. To add a new preprocessing module:

1. **Implement the Preprocessor:** Create a new class in the `preprocessing/` directory (or a new file if appropriate). This class should inherit from the base `Preprocessing` class (defined in `preprocessing.py`). Implement the required methods, such as `apply()` or `transform()`, to perform your preprocessing on the EEG data (for example, a new filter or feature computation).
2. **Register the Preprocessor:** Ensure your new class is imported or registered in `preprocessing.py`. Similar to models, if the config refers to a preprocessing by name, you might need to add an entry so that name maps to your class. Also, create a YAML config in `config/preprocessing/` for your preprocessing parameters if needed.
3. **Integrate into Config:** Modify the experiment config (run config YAML) to use your preprocessing. For example, set `preprocessing: my_new_preproc` in the YAML or override it via command line (e.g., `preprocessing=my_new_preproc`). This will ensure that when `main.py` runs, it constructs your preprocessing and applies it to the data.

## Outputs and Submission

After training completes, the pipeline will automatically generate a **submission file** with the model’s predictions on the test set. The `main.py` (in conjunction with `helper/submission.py`) handles this, typically right after the final evaluation. You do *not* need to run a separate inference script – the best model checkpoint (as determined by validation performance) is used to produce the output.

The submission file is usually saved in the output directory of the run. By default (with Hydra), each run creates a unique folder under `outputs/` (or `multirun/` for multi-runs). Inside that folder, you will find the submission results file (for example, a CSV file containing the predicted labels or event timings). The exact naming vary (e.g., `$date_time_of_training$.csv`), but it will be generated automatically at the end of training.