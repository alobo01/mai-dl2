#!/usr/bin/env python3
"""
Simple test script to validate CPU training setup.
"""

import sys
import os
import torch

def test_basic_imports():
    """Test basic imports."""
    try:
        print("Testing basic imports...")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Python version: {sys.version}")
        
        # Test hydra
        import hydra
        print(f"Hydra imported successfully")
        
        # Test pytorch lightning
        import pytorch_lightning as pl
        print(f"PyTorch Lightning version: {pl.__version__}")
        
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    try:
        print("\nTesting config loading...")
        from omegaconf import DictConfig, OmegaConf
        
        # Test loading a basic config
        config_path = "configs/exp_cifar10_threshold_1u.yaml"
        if os.path.exists(config_path):
            cfg = OmegaConf.load(config_path)
            print(f"Config loaded: {config_path}")
            print(f"Experiment name: {cfg.experiment.name}")
            print("✅ Config loading successful")
            return True
        else:
            print(f"❌ Config file not found: {config_path}")
            return False
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False

def test_module_imports():
    """Test deep_osr module imports."""
    try:
        print("\nTesting deep_osr imports...")
        
        # Add src to path
        src_path = os.path.join(os.path.dirname(__file__), 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from deep_osr.utils.seed import seed_everything
        print("✅ Seed utils imported")
        
        from deep_osr.data.dataset import OpenSetDataModule
        print("✅ Data module imported")
        
        from deep_osr.train_module import OpenSetLightningModule
        print("✅ Train module imported")
        
        # Test enhanced modules
        try:
            from deep_osr.enhanced_data_module import EnhancedOpenSetDataModule
            from deep_osr.enhanced_train_module import EnhancedOpenSetLightningModule
            print("✅ Enhanced modules imported")
            ENHANCED_AVAILABLE = True
        except ImportError as e:
            print(f"⚠️ Enhanced modules not available: {e}")
            ENHANCED_AVAILABLE = False
        
        return True
    except Exception as e:
        print(f"❌ Module import error: {e}")
        return False

def test_cpu_trainer():
    """Test creating a CPU trainer."""
    try:
        print("\nTesting CPU trainer creation...")
        import pytorch_lightning as pl
        
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=False,
            limit_train_batches=1,
            limit_val_batches=1
        )
        print("✅ CPU trainer created successfully")
        return True
    except Exception as e:
        print(f"❌ CPU trainer error: {e}")
        return False

if __name__ == "__main__":
    print("=== CPU Training Test ===")
    
    all_passed = True
    all_passed &= test_basic_imports()
    all_passed &= test_config_loading()
    all_passed &= test_module_imports()
    all_passed &= test_cpu_trainer()
    
    if all_passed:
        print("\n🎉 All tests passed! CPU training should work.")
    else:
        print("\n💥 Some tests failed. Check the errors above.")
    
    print("\nNow testing actual training command...")
    print("Command: python -m src.deep_osr.train --config-name exp_cifar10_threshold_1u train.trainer.max_epochs=1 train.trainer.limit_train_batches=2 train.trainer.limit_val_batches=1 train.trainer.gpus=0 dataset.num_workers=0")
