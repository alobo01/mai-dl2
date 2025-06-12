#!/usr/bin/env python3
"""
Quick smoke test for both datasets with 1 epoch each.
Tests one configuration from each dataset to validate the setup.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_smoke_test(config_name, dataset_name):
    """Run a single smoke test experiment."""
    print(f"\n{'='*50}")
    print(f"Testing {dataset_name} - {config_name}")
    print(f"{'='*50}")
    cmd = [
        sys.executable, "-m", "src.deep_osr.train",
        "--config-name", config_name,
        "train.trainer.max_epochs=1",
        "train.trainer.limit_train_batches=10", 
        "train.trainer.limit_val_batches=5",
        "train.trainer.enable_progress_bar=True",
        "train.trainer.logger=False",
        "train.trainer.enable_checkpointing=False",
        "dataset.num_workers=0",
        f"custom_output_dir=smoke_test_outputs/{dataset_name}_{config_name}"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {dataset_name} test PASSED in {duration:.1f}s")
            return True
        else:
            print(f"❌ {dataset_name} test FAILED")
            print("STDOUT:", result.stdout[-500:])
            print("STDERR:", result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {dataset_name} test TIMED OUT")
        return False
    except Exception as e:
        print(f"💥 {dataset_name} test ERROR: {e}")
        return False

def main():
    """Run smoke tests for both datasets."""
    print("Enhanced OSR Dual Dataset Smoke Test")
    print("Running 1 epoch tests for CIFAR-10 and GTSRB")
    
    # Create output directory
    Path("smoke_test_outputs").mkdir(exist_ok=True)
    
    # Test configurations
    tests = [
        ("exp_cifar10_threshold_1u", "CIFAR-10"),
        ("exp_gtsrb_threshold_3u", "GTSRB")
    ]
    
    results = []
    overall_start = time.time()
    
    for config_name, dataset_name in tests:
        success = run_smoke_test(config_name, dataset_name)
        results.append((dataset_name, success))
    
    total_time = time.time() - overall_start
    
    # Summary
    print(f"\n{'='*50}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*50}")
    
    all_passed = True
    for dataset_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{dataset_name}: {status}")
        all_passed = all_passed and success
    
    print(f"\nTotal time: {total_time:.1f}s")
    
    if all_passed:
        print("\n🎉 ALL SMOKE TESTS PASSED!")
        print("You're ready to run full experiments.")
    else:
        print("\n💥 SOME TESTS FAILED!")
        print("Please check the configuration and try again.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
