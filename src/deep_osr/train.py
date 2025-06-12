import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import os
import json
import subprocess # For git hash
import torch
import pandas as pd
import numpy as np
import hashlib

from deep_osr.utils.seed import seed_everything
from deep_osr.data.dataset import OpenSetDataModule
from deep_osr.train_module import OpenSetLightningModule

# Enhanced modules for enhanced OSR experiments
try:
    from deep_osr.enhanced_data_module import EnhancedOpenSetDataModule
    from deep_osr.enhanced_train_module import EnhancedOpenSetLightningModule
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "git_not_available"

def save_test_predictions_csv(model, datamodule, run_dir):
    """Save test predictions to CSV file for later analysis."""
    print("Saving test predictions to CSV...")
    
    model.eval()
    test_loader = datamodule.test_dataloader()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_osr_scores = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels = batch[0], batch[1]
            
            images = images.to(model.device)
              # Forward pass
            outputs = model(images)
            
            if isinstance(outputs, dict):
                logits = outputs.get("cls_logits", outputs.get("logits"))
                osr_scores = outputs.get("osr_score", torch.zeros(images.size(0)))
            elif isinstance(outputs, tuple) and len(outputs) == 3:
                # Enhanced module returns (closed_set_logits_k, dummy_logits, osr_output)
                logits, dummy_logits, osr_output = outputs
                # Get standardized OSR scores
                if hasattr(model, '_get_osr_score_from_outputs'):
                    osr_scores = model._get_osr_score_from_outputs(logits, dummy_logits, osr_output)
                else:
                    osr_scores = osr_output if osr_output is not None else torch.zeros(images.size(0), device=images.device)
            else:
                logits = outputs
                osr_scores = torch.zeros(images.size(0), device=images.device)
            
            # Calculate probabilities and predictions
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_osr_scores.extend(osr_scores.cpu().numpy())
    
    # Create DataFrame
    prob_columns = [f'prob_class_{i}' for i in range(len(all_probabilities[0]))]
    
    df_data = {
        'true_label': all_labels,
        'predicted_label': all_predictions,
        'osr_score': all_osr_scores
    }
    
    # Add probability columns
    probs_array = np.array(all_probabilities)
    for i, col in enumerate(prob_columns):
        df_data[col] = probs_array[:, i]
    
    df = pd.DataFrame(df_data)
    
    # Save to CSV
    csv_path = os.path.join(run_dir, "test_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"Test predictions saved to {csv_path}")
    
    return csv_path

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Handle seed access with fallback for experiment configs
    print(cfg)
    seed = cfg.get("seed", 42)  # Default to 42 if seed is not accessible # Changed
    seed_everything(seed) # Changed: Align with train_classic.py
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    # Config hashing (aligning with train_classic.py)
    resolved_config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    config_hash_full = hashlib.sha256(resolved_config_yaml.encode('utf-8')).hexdigest()
    short_config_hash = config_hash_full[:8]
    
    # Custom output directory logic
    if cfg.get("custom_output_dir"):
        run_dir = cfg.custom_output_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        # Aligning run directory structure with train_classic.py's style
        dataset_name = cfg.dataset.name # Direct access, assuming it exists as in train_classic.py
        # Construct path relative to Hydra's output directory structure
        run_dir = os.path.join(hydra_cfg.runtime.output_dir, "..", "..", "runs", f"{dataset_name}-{short_config_hash}")
        run_dir = os.path.abspath(run_dir) # Get absolute path
        os.makedirs(run_dir, exist_ok=True)
    
    print(f"Run directory: {run_dir}")
    print(f"Full config:\\n{OmegaConf.to_yaml(cfg)}")

    # Save config and git hash
    with open(os.path.join(run_dir, "config_resolved.yaml"), 'w') as f:
        f.write(resolved_config_yaml) # Use the resolved_config_yaml from hashing
    
    meta_data = {
        "git_hash": get_git_hash(),
        "hydra_run_dir_default": hydra_cfg.run.dir, # Hydra's default output for this specific job
        "custom_run_dir": run_dir, # Our calculated run_dir
        "command": " ".join(hydra_cfg.job.override_dirname.split(os.path.sep)) if hydra_cfg.job.override_dirname else "base_run",
        "config_hash": config_hash_full # The full sha256 hash
    }
    with open(os.path.join(run_dir, "meta.json"), 'w') as f:
        json.dump(meta_data, f, indent=2)
    # Data module - check if enhanced version should be used
    use_enhanced = ENHANCED_AVAILABLE and cfg.get("experiment", {}).get("use_enhanced", False)
    
    if use_enhanced:
        print("Using Enhanced OpenSet modules")
        datamodule = EnhancedOpenSetDataModule(cfg.dataset)
        model = EnhancedOpenSetLightningModule(cfg)
    else:
        print("Using standard OpenSet modules")
        datamodule = OpenSetDataModule(cfg.dataset)
        model = OpenSetLightningModule(cfg)

    # Callbacks
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename='{epoch}-{val/acc_known:.3f}-{val/loss_total:.3f}', # Changed val/auroc to val/acc_known
        monitor='val/acc_known',  # Changed val/auroc to val/acc_known
        mode='max',
        save_top_k=1,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
      # Add early stopping callback (enabled by default for HTCondor jobs)
    early_stop_callback = EarlyStopping(
        monitor='val/acc_known',  # Monitor validation accuracy as requested
        patience=10, 
        verbose=True, 
        mode='max'
    )    
    callbacks.append(early_stop_callback)
    
    # Only add LearningRateMonitor if a logger is enabled
    # (LearningRateMonitor requires a logger to function)
    enable_logger = cfg.train.trainer.get("logger", False)
    if enable_logger:
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
    
    # Trainer - Handle GPU/CPU selection properly
    gpus = cfg.train.trainer.get("gpus", 1)
    use_cpu = gpus == 0
    
    trainer = pl.Trainer(
        max_epochs=cfg.train.trainer.max_epochs,
        accelerator="cpu" if use_cpu else "gpu",
        devices=1 if use_cpu else gpus,
        precision=cfg.train.trainer.precision,
        callbacks=callbacks,
        logger=enable_logger,  # Use config setting for logger
        deterministic=cfg.train.trainer.get("deterministic", True),
        # Batch limiting for quick testing
        limit_train_batches=cfg.train.trainer.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.train.trainer.get("limit_val_batches", 1.0),
        limit_test_batches=cfg.train.trainer.get("limit_test_batches", 1.0),
        # Progress and logging settings
        enable_progress_bar=cfg.train.trainer.get("enable_progress_bar", True),
        enable_model_summary=cfg.train.trainer.get("enable_model_summary", True),
        enable_checkpointing=cfg.train.trainer.get("enable_checkpointing", True),
        default_root_dir=run_dir
    )

    trainer.fit(model, datamodule=datamodule)

    # After training, perform OpenMax MAV fitting if applicable
    if model.osr_head_type == "openmax":
        print("Fitting OpenMax MAVs using the trained model (best checkpoint)...")
        # Load best checkpoint to ensure MAVs are fitted on the best model state
        best_model_path = checkpoint_callback.best_model_path
        if not best_model_path: # If no best model (e.g. validation sanity check failed or training too short)
            print("No best model path found, using current model state for MAV fitting.")
            model_for_mav_fitting = model
        else:
            print(f"Loading best model from: {best_model_path}")
            if use_enhanced:
                model_for_mav_fitting = EnhancedOpenSetLightningModule.load_from_checkpoint(best_model_path, cfg=cfg)
            else:
                model_for_mav_fitting = OpenSetLightningModule.load_from_checkpoint(best_model_path, cfg=cfg) # cfg needed if not in hparams
        
        model_for_mav_fitting.to(model.device) # Ensure it's on the same device
        datamodule.setup(stage='fit') # Ensure train_dataloader is ready for MAV fitting
        model_for_mav_fitting.fit_openmax_if_needed(datamodule)
        # Save the model state again if MAVs were fitted and are part of the model's state (e.g. buffers)
        # This is important if OpenMaxHead stores MAVs.
        # If OpenMaxHead is used, the best checkpoint should be updated/resaved after MAV fitting.
        # For simplicity, let's assume eval.py will handle MAV fitting if OpenMax is chosen, or
        # here we save a separate checkpoint like `ckpt_best_with_mavs.pt`
        # trainer.save_checkpoint(os.path.join(run_dir, "checkpoints", "last_with_mavs.ckpt")) # Example
        # For now, the MAVs are part of the OpenMaxOSRHead's state. The loaded model_for_mav_fitting
        # will have them. We might need to save this updated model.
        # The `eval.py` will need to load the checkpoint and then call fit_mavs.
        # OR, train.py saves the MAVs separately, and eval.py loads them.
        # Let's assume `eval.py` will handle MAV fitting on the loaded checkpoint.
        # This simplifies `train.py`. So, remove MAV fitting from here.
        pass


    # After training, perform calibration if specified
    if cfg.train.calibration.method:
        print("Calibrating model using the trained model (best checkpoint)...")
        best_model_path = checkpoint_callback.best_model_path
        if not best_model_path :
            print("No best model path found, using current model state for calibration.")
            model_for_calibration = model
        else:
            print(f"Loading best model from: {best_model_path}")
            if use_enhanced:
                model_for_calibration = EnhancedOpenSetLightningModule.load_from_checkpoint(best_model_path, cfg=cfg)
            else:
                model_for_calibration = OpenSetLightningModule.load_from_checkpoint(best_model_path, cfg=cfg)

        model_for_calibration.to(model.device)
        datamodule.setup(stage='validate') # Ensure calibration_dataloader (val_dataloader) is ready
        model_for_calibration.calibrate_temperature(datamodule)
        
        # Save the calibration temperature. It can be saved within the model checkpoint,
        # or as a separate file. eval.py will need this.        # For simplicity, let's save it to a file in the run_dir.
        calibration_info = {
            'method': cfg.train.calibration.method,
            'temperature': model_for_calibration.calibration_temperature if hasattr(model_for_calibration, 'calibration_temperature') else 1.0
        }
        with open(os.path.join(run_dir, "calibration_info.json"), 'w') as f:
            json.dump(calibration_info, f, indent=2)
        print(f"Calibration info saved to {os.path.join(run_dir, 'calibration_info.json')}")

    # Run test set evaluation and save predictions
    print("Running test set evaluation...")
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        print("No best model path found, using current model state for testing.")
        model_for_test = model
    else:
        print(f"Loading best model from: {best_model_path}")
        if use_enhanced:
            # Pass cfg to ensure correct model instantiation
            model_for_test = EnhancedOpenSetLightningModule.load_from_checkpoint(best_model_path, cfg=cfg)
        else:
            # Pass cfg to ensure correct model instantiation
            model_for_test = OpenSetLightningModule.load_from_checkpoint(best_model_path, cfg=cfg)

    model_for_test.to(model.device)
    datamodule.setup(stage='test')
    
    # Run test and save predictions
    test_results = trainer.test(model_for_test, datamodule=datamodule, verbose=False)
    
    # Save test predictions as CSV
    save_test_predictions_csv(model_for_test, datamodule, run_dir)

    print(f"Training finished. Checkpoints and logs saved in: {run_dir}")

if __name__ == "__main__":
    main()