import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import os
import json
import subprocess # For git hash

from deep_osr.utils.seed import seed_everything
from deep_osr.data.dataset import OpenSetDataModule
from deep_osr.train_module import OpenSetLightningModule

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "git_not_available"

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    run_dir = hydra_cfg.run.dir # Hydra's current run directory (e.g., outputs/runs/YYYY-MM-DD_HH-MM-SS)
    print(f"Hydra run directory: {run_dir}")
    print(f"Full config:\n{OmegaConf.to_yaml(cfg)}")

    # Save config and git hash
    with open(os.path.join(run_dir, "config_resolved.yaml"), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    
    meta_data = {
        "git_hash": get_git_hash(),
        "hydra_run_dir": run_dir,
        "command": " ".join(hydra_cfg.job.override_dirname.split(os.path.sep)) if hydra_cfg.job.override_dirname else "base_run" # TODO better way to get command
    }
    with open(os.path.join(run_dir, "meta.json"), 'w') as f:
        json.dump(meta_data, f, indent=2)

    # Data module
    datamodule = OpenSetDataModule(cfg.dataset)
    # datamodule.prepare_data() # Called by trainer
    # datamodule.setup()      # Called by trainer

    # Model
    model = OpenSetLightningModule(cfg)

    # Callbacks
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename='{epoch}-{val/auroc:.3f}-{val/acc_known:.3f}',
        monitor='val/auroc', 
        mode='max',
        save_top_k=1,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    if cfg.train.trainer.get("enable_early_stopping", False): # Example for early stopping
        early_stop_callback = EarlyStopping(
            monitor='val/auroc', patience=10, verbose=True, mode='max'
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
        logger=pl.loggers.TensorBoardLogger(save_dir=run_dir, name="tb_logs", version=""), # Logs to <run_dir>/tb_logs
        deterministic=cfg.train.trainer.get("deterministic", True),
        # check_val_every_n_epoch=cfg.train.trainer.check_val_every_n_epoch, # Default is 1
        # Can add profiler etc.
        default_root_dir=run_dir # Ensure logs/outputs related to PL are within Hydra's dir
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
            model_for_calibration = OpenSetLightningModule.load_from_checkpoint(best_model_path, cfg=cfg)

        model_for_calibration.to(model.device)
        datamodule.setup(stage='validate') # Ensure calibration_dataloader (val_dataloader) is ready
        model_for_calibration.calibrate_temperature(datamodule)
        
        # Save the calibration temperature. It can be saved within the model checkpoint,
        # or as a separate file. eval.py will need this.
        # For simplicity, let's save it to a file in the run_dir.
        calibration_info = {
            'method': cfg.train.calibration.method,
            'temperature': model_for_calibration.calibration_temperature if hasattr(model_for_calibration, 'calibration_temperature') else 1.0
        }
        with open(os.path.join(run_dir, "calibration_info.json"), 'w') as f:
            json.dump(calibration_info, f, indent=2)
        print(f"Calibration info saved to {os.path.join(run_dir, 'calibration_info.json')}")


    # Optional: Run test set evaluation
    # trainer.test(model, datamodule=datamodule, ckpt_path='best') # Loads best checkpoint

    print(f"Training finished. Checkpoints and logs saved in: {run_dir}")

if __name__ == "__main__":
    main()