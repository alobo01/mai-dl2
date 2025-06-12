#!/usr/bin/env python3
"""
Quick test launcher for enhanced OSR experiments.
This script runs a single experiment to validate the implementation.
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def run_test_experiment():
    """Run a quick test experiment to validate the implementation."""
    
    print("=" * 60)
    print("Enhanced OSR Implementation Test")
    print("=" * 60)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print(f"Working directory: {project_dir}")
    print(f"Python executable: {sys.executable}")
    
    # Test 1: Check if modules can be imported
    print("\n1. Testing module imports...")
    
    try:
        from src.deep_osr.losses.enhanced_nll import ThresholdedNLLLoss, DummyClassNLLLoss, CombinedThresholdPenaltyLoss
        from src.deep_osr.enhanced_train_module import EnhancedOpenSetLightningModule
        from src.deep_osr.enhanced_data_module import EnhancedOpenSetDataModule
        print("✓ All enhanced modules imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test 2: Validate configuration files
    print("\n2. Validating configuration files...")
    
    config_files = [
        "configs/exp_cifar10_threshold_1u.yaml",
        "configs/model/resnet50_enhanced_osr.yaml",
        "configs/train/enhanced_osr.yaml"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✓ {config_file}")
        else:
            print(f"✗ Missing: {config_file}")
            return False
    
    # Test 3: Quick training test (1 epoch)
    print("\n3. Running quick training test...")
    
    test_cmd = [
        sys.executable, "-m", "src.train",
        "--config-path", "configs",
        "--config-name", "exp_cifar10_threshold_1u",
        "trainer.max_epochs=1",
        "trainer.limit_train_batches=10",
        "trainer.limit_val_batches=5",
        "trainer.enable_progress_bar=False",
        "trainer.logger=False",
        "trainer.enable_checkpointing=False",
        "dataset.num_workers=0"  # Avoid multiprocessing issues
    ]
    
    try:
        print("Running command:", " ".join(test_cmd))
        result = subprocess.run(
            test_cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✓ Quick training test passed")
            return True
        else:
            print("✗ Training test failed")
            print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
            print("STDERR:", result.stderr[-1000:])
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Training test timed out")
        return False
    except Exception as e:
        print(f"✗ Training test error: {e}")
        return False

def run_full_experiment():
    """Run a complete single experiment for validation."""
    
    print("\n" + "=" * 60)
    print("Running Full Single Experiment")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("test_experiment_results")
    results_dir.mkdir(exist_ok=True)
    
    # Run single experiment
    cmd = [
        sys.executable, "run_enhanced_experiments.py",
        "--config-dir", "configs",
        "--dataset", "cifar10",
        "--gpu-ids", "0"
    ]
    
    print("Running enhanced experiments...")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, timeout=3600)  # 1 hour timeout
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Full experiment completed in {duration:.1f}s")
            return True
        else:
            print(f"✗ Full experiment failed (return code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Full experiment timed out")
        return False
    except Exception as e:
        print(f"✗ Full experiment error: {e}")
        return False

def main():
    """Main test runner."""
    
    print("Enhanced OSR Implementation Validation")
    print("======================================\n")
    
    # Step 1: Quick validation test
    if not run_test_experiment():
        print("\n❌ Quick test failed. Please check the implementation.")
        sys.exit(1)
    
    print("\n✅ Quick test passed!")
    
    # Ask user if they want to run full experiment
    response = input("\nRun a full experiment? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        if run_full_experiment():
            print("\n✅ Full experiment completed successfully!")
            print("\nNext steps:")
            print("1. Check results in 'enhanced_experiment_results/' directory")
            print("2. Run analysis: python analyze_enhanced_results.py")
            print("3. Execute all experiments: python run_enhanced_experiments.py")
        else:
            print("\n❌ Full experiment failed. Check logs for details.")
    else:
        print("\n✅ Validation complete!")
        print("\nTo run experiments:")
        print("- Single experiment: python run_enhanced_experiments.py --dataset cifar10")
        print("- All experiments: python run_enhanced_experiments.py")
        print("- Analysis: python analyze_enhanced_results.py")

if __name__ == "__main__":
    main()
