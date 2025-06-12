import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import os
import json
import subprocess # For git hash
import hashlib

from deep_osr.utils.seed import seed_everything
from deep_osr.data.classic_dataset import ClassicDataModule # Changed import
from deep_osr.classic_train_module import ClassicLightningModule # Changed import
from deep_osr.utils.evaluation_utils import ClassicEvaluator # New import

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "git_not_available"

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    
    # --- Custom Run Directory --- #
    resolved_config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    config_hash = hashlib.sha256(resolved_config_yaml.encode('utf-8')).hexdigest()
    short_config_hash = config_hash[:8] # Use a short hash for brevity
    
    # Construct the custom run directory path
    # Note: Hydra's default output dir is outputs/YYYY-MM-DD/HH-MM-SS or outputs/multirun/...
    # We are creating a specific directory structure under 'outputs/runs/'
    # Ensure the base 'outputs/runs' directory exists if it's the root for these custom runs.
    # os.makedirs("outputs/runs", exist_ok=True) # Base directory for all runs
    # The custom_run_dir will be like: outputs/runs/cifar10-abcdef12
    custom_run_dir = os.path.join(hydra_cfg.runtime.output_dir, "..", "..", "runs", f"{cfg.dataset.name}-{short_config_hash}")
    custom_run_dir = os.path.abspath(custom_run_dir) # Get absolute path
    os.makedirs(custom_run_dir, exist_ok=True)

    print(f"Hydra default run directory: {hydra_cfg.run.dir}")
    print(f"Custom run directory: {custom_run_dir}")
    print(f"Full config: {OmegaConf.to_yaml(cfg)}")

    # Save config and git hash to custom run directory
    with open(os.path.join(custom_run_dir, "config_resolved.yaml"), 'w') as f:
        f.write(resolved_config_yaml)
    
    meta_data = {
        "git_hash": get_git_hash(),
        "hydra_run_dir_default": hydra_cfg.run.dir,
        "custom_run_dir": custom_run_dir,
        "command": " ".join(hydra_cfg.job.override_dirname.split(os.path.sep)) if hydra_cfg.job.override_dirname else "base_run",
        "config_hash": config_hash
    }
    with open(os.path.join(custom_run_dir, "meta.json"), 'w') as f:
        json.dump(meta_data, f, indent=2)

    # Data module
    datamodule = ClassicDataModule(cfg.dataset)

    # Model
    model = ClassicLightningModule(cfg)

    # Callbacks
    callbacks = []
    # Checkpoint callback for classic training (monitor val_acc or val_f1_macro)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(custom_run_dir, "checkpoints"),
        filename='{epoch}-{val/acc:.3f}-{val/loss:.3f}', # Adjusted filename
        monitor='val/acc', 
        mode='max',
        save_top_k=1,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    if cfg.train.trainer.get("enable_early_stopping", False):
        early_stop_callback = EarlyStopping(
            monitor=cfg.train.trainer.get("early_stopping_monitor", 'val/acc'), # Adjusted monitor
            patience=cfg.train.trainer.get("early_stopping_patience", 10),
            verbose=True, 
            mode=cfg.train.trainer.get("early_stopping_mode", 'max') # Adjusted mode
        )
        callbacks.append(early_stop_callback)
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.trainer.max_epochs,
        accelerator="gpu" if cfg.train.trainer.gpus > 0 else "cpu",
        devices=cfg.train.trainer.gpus if cfg.train.trainer.gpus > 0 else 1,
        precision=cfg.train.trainer.precision,
        callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger(save_dir=custom_run_dir, name="tb_logs", version=""), 
        deterministic=cfg.train.trainer.get("deterministic", True),
        default_root_dir=custom_run_dir 
    )

    trainer.fit(model, datamodule=datamodule)

    # Optional: Run test set evaluation using the best checkpoint
    if cfg.train.get("run_test_after_train", True):
        print("Running standard test set evaluation (on knowns with mapped labels)...")
        # The standard test_dataloader from ClassicDataModule should provide knowns with mapped labels
        # if the model was trained on mapped labels.
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Loading best model from: {best_model_path}")
            trainer.test(model, datamodule=datamodule, ckpt_path=best_model_path)
        else:
            print("No best model path found, testing with last model state.")
            trainer.test(model, datamodule=datamodule)

    # Custom Evaluation Step
    if cfg.train.get("run_custom_evaluation_after_train", False):
        print("Running custom evaluation on validation set (knowns and unknowns)...")
        best_model_path = checkpoint_callback.best_model_path
        if not best_model_path:
            print("Warning: No best model checkpoint found for custom evaluation. Using last model state.")
            eval_model = model
        else:
            print(f"Loading best model from {best_model_path} for custom evaluation.")
            # Ensure the model loaded is the same type and configured correctly
            eval_model = ClassicLightningModule.load_from_checkpoint(best_model_path, cfg=cfg) 

        evaluator = ClassicEvaluator(eval_model, datamodule, cfg, custom_run_dir)
        evaluator.evaluate_and_save_results()

    print(f"Training and evaluation finished. Checkpoints, logs, and evaluation results saved in: {custom_run_dir}")

if __name__ == "__main__":
    main()
