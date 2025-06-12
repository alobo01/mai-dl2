#!/usr/bin/env python3
"""
HTCondor Readiness Checker for Enhanced OSR Experiments
======================================================

This script checks if all files and configurations are ready for HTCondor submission.
"""

import os
import sys
from pathlib import Path
import yaml

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} - MISSING")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists and report status."""
    if Path(dirpath).is_dir():
        print(f"✓ {description}: {dirpath}")
        return True
    else:
        print(f"✗ {description}: {dirpath} - MISSING")
        return False

def check_config_files():
    """Check all experiment configuration files."""
    configs_dir = Path("configs")
    
    # Expected experiment configs
    expected_configs = [
        "exp_cifar10_threshold_1u.yaml",
        "exp_cifar10_penalty_3u.yaml", 
        "exp_cifar10_combined_5u.yaml",
        "exp_cifar10_high_penalty_1u.yaml",
        "exp_cifar10_high_threshold_3u.yaml",
        "exp_cifar10_optimized_combined_5u.yaml",
        "exp_gtsrb_threshold_3u.yaml",
        "exp_gtsrb_penalty_6u.yaml",
        "exp_gtsrb_combined_9u.yaml",
        "exp_gtsrb_high_penalty_3u.yaml",
        "exp_gtsrb_low_threshold_6u.yaml",
        "exp_gtsrb_optimized_combined_9u.yaml"
    ]
    
    print("\nChecking experiment configurations:")
    all_present = True
    
    for config in expected_configs:
        config_path = configs_dir / config
        if config_path.exists():
            # Validate config can be loaded
            try:
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                print(f"✓ {config}")
            except Exception as e:
                print(f"✗ {config} - INVALID YAML: {e}")
                all_present = False
        else:
            print(f"✗ {config} - MISSING")
            all_present = False
    
    return all_present

def check_dataset_configs():
    """Check dataset configuration files."""
    dataset_configs = [
        "configs/dataset/cifar10_1u.yaml",
        "configs/dataset/cifar10_3u.yaml",
        "configs/dataset/cifar10_5u.yaml"
    ]
    
    print("\nChecking dataset configurations:")
    all_present = True
    
    for config in dataset_configs:
        if not check_file_exists(config, f"Dataset config"):
            all_present = False
    
    return all_present

def check_source_code():
    """Check source code files."""
    source_files = [
        "src/deep_osr/train.py",
        "src/deep_osr/enhanced_train_module.py",
        "src/deep_osr/enhanced_data_module.py",
        "src/deep_osr/losses/enhanced_nll.py"
    ]
    
    print("\nChecking source code:")
    all_present = True
    
    for source_file in source_files:
        if not check_file_exists(source_file, "Source file"):
            all_present = False
    
    return all_present

def check_htcondor_files():
    """Check HTCondor submission files."""
    htcondor_files = [
        "htcondor_job.sh",
        "enhanced_osr_experiments.sub",
        "smoke_test.sub"
    ]
    
    print("\nChecking HTCondor files:")
    all_present = True
    
    for htc_file in htcondor_files:
        if not check_file_exists(htc_file, "HTCondor file"):
            all_present = False
    
    # Check if shell script is executable (on Unix systems)
    if os.name != 'nt':  # Not Windows
        htc_script = Path("htcondor_job.sh")
        if htc_script.exists() and not os.access(htc_script, os.X_OK):
            print(f"⚠ htcondor_job.sh exists but is not executable")
            print("  Run: chmod +x htcondor_job.sh")
    
    return all_present

def check_data_directories():
    """Check data directories."""
    data_dirs = [
        "data",
        "data/cifar-10-batches-py",
        "data/gtsrb"
    ]
    
    print("\nChecking data directories:")
    all_present = True
    
    for data_dir in data_dirs:
        if not check_directory_exists(data_dir, "Data directory"):
            all_present = False
    
    return all_present

def check_python_imports():
    """Check if required Python modules can be imported."""
    required_modules = [
        "torch",
        "pytorch_lightning", 
        "hydra",
        "omegaconf",
        "pandas",
        "numpy",
        "sklearn"
    ]
    
    print("\nChecking Python dependencies:")
    all_available = True
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - NOT AVAILABLE")
            all_available = False
    
    return all_available

def create_logs_directory():
    """Create logs directory for HTCondor."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        logs_dir.mkdir()
        print(f"✓ Created logs directory: {logs_dir}")
    else:
        print(f"✓ Logs directory exists: {logs_dir}")

def main():
    """Main readiness check."""
    print("HTCondor Enhanced OSR Experiments Readiness Check")
    print("=" * 50)
    
    # Create necessary directories
    create_logs_directory()
    
    # Run all checks
    checks = [
        ("HTCondor files", check_htcondor_files),
        ("Source code", check_source_code),
        ("Experiment configurations", check_config_files),
        ("Dataset configurations", check_dataset_configs),
        ("Data directories", check_data_directories),
        ("Python dependencies", check_python_imports)
    ]
    
    all_passed = True
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
            all_passed = all_passed and results[check_name]
        except Exception as e:
            print(f"✗ Error during {check_name} check: {e}")
            results[check_name] = False
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    print("READINESS SUMMARY")
    print("=" * 50)
    
    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        icon = "✓" if passed else "✗"
        print(f"{icon} {check_name}: {status}")
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("\nYou are ready to submit HTCondor jobs:")
        print("1. Smoke test: condor_submit smoke_test.sub")
        print("2. Full experiments: condor_submit enhanced_osr_experiments.sub")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("\nPlease fix the issues above before submitting to HTCondor.")
        print("\nCommon fixes:")
        print("- Install missing Python packages: pip install <package>")
        print("- Create missing config files")
        print("- Download/prepare datasets")
        print("- Make shell script executable: chmod +x htcondor_job.sh")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
