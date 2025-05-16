# Deep Open-Set Recognition (OSR) Framework

This repository provides a PyTorch Lightning and Hydra based framework for training and evaluating deep learning models for open-set recognition.

## Features

- Modular design: Backbone, Neck, Classifier Head, OSR Head.
- Supports various backbones (ResNet, EfficientNet, ViT, DINOv2).
- Multiple OSR head strategies: Energy-based, OpenMax (simplified), K+1 Softmax.
- Configuration via Hydra (YAML files).
- Standardized evaluation pipeline for unseen classes.
- Scripts for training, evaluation, and generating plots (OSCR, ROC, t-SNE, Confusion Matrix).
- Scalable repository layout.

## Project Structure

```
deep-osr/
├─ configs/         # Hydra configuration files (dataset, model, train, eval, experiments)
├─ data/            # Data (not committed, .gitignored)
├─ src/             # Source code
│  ├─ data/          # Dataset and dataloader logic
│  ├─ models/        # Model components (backbone, neck, heads)
│  ├─ losses/        # Loss functions
│  ├─ utils/         # Utility functions (seeding, calibration)
│  ├─ visualize/     # Plotting scripts
│  ├─ open_set_metrics.py # OSR metric calculation logic
│  ├─ train_module.py   # PyTorch Lightning module
│  ├─ train.py       # Training script
│  └─ eval.py        # Evaluation script
├─ scripts/         # Shell scripts for workflows
├─ outputs/         # Default output directory for Hydra runs (logs, checkpoints, metrics, plots)
└─ ... (other standard folders: tests, docs, etc.)
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd deep-osr
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install `src` as a package (for easier imports and script execution):**
    ```bash
    pip install -e .
    ```

## Dataset

The example uses CIFAR10. It will be automatically downloaded by `torchvision` to `data/processed/cifar10` (or as specified by `data_root_dir` in `configs/config.yaml`).
The default configuration `configs/dataset/cifar10_8k_2u.yaml` uses classes 0-7 as known and 8-9 as unknown.

## Running the Example Workflow

### 1. Train a Model

Use `scripts/run_train.sh` with an experiment configuration. An example experiment `cifar10_resnet50_energy` is provided.

```bash
# From the deep-osr root directory:
scripts/run_train.sh experiment/cifar10_resnet50_energy
```
This will:
- Read configurations from `configs/experiment/cifar10_resnet50_energy.yaml` (which composes other base configs).
- Start training using `src/train.py`.
- Save logs, checkpoints, and configs into `outputs/runs/YYYY-MM-DD_HH-MM-SS/`.
- After training, it will perform temperature calibration and save `calibration_info.json`.

You can override any configuration parameter from the command line:
```bash
scripts/run_train.sh experiment/cifar10_resnet50_energy train.trainer.max_epochs=10 model.backbone.name=efficientnet_b4
```

To run a sweep (multirun):
```bash
python -m src.train -m model.backbone.name=resnet50,vit_b16 model.osr_head.type=energy,kplus1 +experiment=experiment/cifar10_resnet50_energy train.trainer.max_epochs=5
```
Check `scripts/sweep.yaml` for more details on Hydra sweeps.

### 2. Evaluate the Trained Model

After training, a run directory (e.g., `outputs/runs/2025-05-16_10-12-00`) will be created. Use `scripts/run_eval.sh` with the path to this run directory.

```bash
# Replace YYYY-MM-DD_HH-MM-SS with your actual run directory name
scripts/run_eval.sh outputs/runs/YYYY-MM-DD_HH-MM-SS 
```
This will:
- Load the best checkpoint and configuration from the specified training run.
- Perform MAV fitting if the model is `openmax`.
- Load calibration temperature.
- Evaluate the model on the test set (known and unknown classes).
- Compute OSR metrics (AUROC, AUPR, U-Recall@95-Seen-Precision, etc.).
- Save metrics to `outputs/runs/YYYY-MM-DD_HH-MM-SS/eval_outputs/metrics_RUN_ID.json`.
- Save raw scores and embeddings (if `eval.save_features_for_tsne=True`) to `outputs/runs/YYYY-MM-DD_HH-MM-SS/eval_outputs/scores_RUN_ID.pkl`.

### 3. Generate Plots

Use `scripts/gen_plots.sh` with the path to the same run directory.

```bash
scripts/gen_plots.sh outputs/runs/YYYY-MM-DD_HH-MM-SS
```
This will:
- Use the `scores_RUN_ID.pkl` file generated during evaluation.
- Generate OSCR curve, ROC curve, t-SNE plot (if features available), and confusion matrix.
- Save plots to `outputs/runs/YYYY-MM-DD_HH-MM-SS/plots/`.

## Configuration Details

- **`configs/config.yaml`**: Main Hydra configuration, sets defaults.
- **`configs/dataset/`**: Dataset specific configurations.
- **`configs/model/`**: Model architecture configurations (backbone, neck, heads).
- **`configs/train/`**: Training parameters (optimizer, scheduler, epochs).
- **`configs/eval/`**: Evaluation parameters.
- **`configs/experiment/`**: Files that compose a full experiment from base dataset, model, and train configs.

## Notes

- **DINOv2**: Using `dino_v2` backbones requires an internet connection for the first time to download from `torch.hub`. Ensure `timm` is installed.
- **OpenMax**: The implemented OpenMax head is simplified (uses MAV distances). Full OpenMax with Weibull fitting is more complex and may require external libraries (like `libmr`). MAV fitting for OpenMax happens during the `eval.py` script, or can be triggered after `train.py` if `OpenSetLightningModule.fit_openmax_if_needed` is called explicitly on the trained model. The current `train.py` defers MAV fitting to `eval.py` for simplicity.
- **Calibration**: Temperature scaling is performed after training if `train.calibration.method=temperature_scaling`. The `calibration_info.json` (containing the temperature) is saved and used by `eval.py`.

This provides a comprehensive setup. Some parts (like the full complexity of OpenMax, advanced calibration methods, or exhaustive unit tests) are simplified for this example but can be extended.