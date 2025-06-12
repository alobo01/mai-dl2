#!/usr/bin/env python3
"""
Script to run all generated experiments
"""
import os
import subprocess
import sys
from pathlib import Path

def run_experiment(config_file):
    """Run a single experiment"""
    cmd = [
        "python", "train_enhanced_osr.py",  # Assuming this is your training script
        "--config", f"configs/experiments/{config_file}"
    ]
    
    print(f"Running experiment: {config_file}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] Completed: {config_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] Failed: {config_file}")
        print(f"Error: {e.stderr}")
        return False

def main():
    experiments = [
        "exp_cifar10_threshold_nonep_lowt_1u_no_neck.yaml",
        "exp_cifar10_threshold_nonep_lowt_1u_neck.yaml",
        "exp_cifar10_threshold_nonep_hight_1u_no_neck.yaml",
        "exp_cifar10_threshold_nonep_hight_1u_neck.yaml",
        "exp_cifar10_penalty_lowp_nonet_1u_no_neck.yaml",
        "exp_cifar10_penalty_lowp_nonet_1u_neck.yaml",
        "exp_cifar10_combined_lowp_lowt_1u_no_neck.yaml",
        "exp_cifar10_combined_lowp_lowt_1u_neck.yaml",
        "exp_cifar10_combined_lowp_hight_1u_no_neck.yaml",
        "exp_cifar10_combined_lowp_hight_1u_neck.yaml",
        "exp_cifar10_penalty_highp_nonet_1u_no_neck.yaml",
        "exp_cifar10_penalty_highp_nonet_1u_neck.yaml",
        "exp_cifar10_combined_highp_lowt_1u_no_neck.yaml",
        "exp_cifar10_combined_highp_lowt_1u_neck.yaml",
        "exp_cifar10_combined_highp_hight_1u_no_neck.yaml",
        "exp_cifar10_combined_highp_hight_1u_neck.yaml",
        "exp_cifar10_threshold_nonep_lowt_3u_no_neck.yaml",
        "exp_cifar10_threshold_nonep_lowt_3u_neck.yaml",
        "exp_cifar10_threshold_nonep_hight_3u_no_neck.yaml",
        "exp_cifar10_threshold_nonep_hight_3u_neck.yaml",
        "exp_cifar10_penalty_lowp_nonet_3u_no_neck.yaml",
        "exp_cifar10_penalty_lowp_nonet_3u_neck.yaml",
        "exp_cifar10_combined_lowp_lowt_3u_no_neck.yaml",
        "exp_cifar10_combined_lowp_lowt_3u_neck.yaml",
        "exp_cifar10_combined_lowp_hight_3u_no_neck.yaml",
        "exp_cifar10_combined_lowp_hight_3u_neck.yaml",
        "exp_cifar10_penalty_highp_nonet_3u_no_neck.yaml",
        "exp_cifar10_penalty_highp_nonet_3u_neck.yaml",
        "exp_cifar10_combined_highp_lowt_3u_no_neck.yaml",
        "exp_cifar10_combined_highp_lowt_3u_neck.yaml",
        "exp_cifar10_combined_highp_hight_3u_no_neck.yaml",
        "exp_cifar10_combined_highp_hight_3u_neck.yaml",
        "exp_cifar10_threshold_nonep_lowt_5u_no_neck.yaml",
        "exp_cifar10_threshold_nonep_lowt_5u_neck.yaml",
        "exp_cifar10_threshold_nonep_hight_5u_no_neck.yaml",
        "exp_cifar10_threshold_nonep_hight_5u_neck.yaml",
        "exp_cifar10_penalty_lowp_nonet_5u_no_neck.yaml",
        "exp_cifar10_penalty_lowp_nonet_5u_neck.yaml",
        "exp_cifar10_combined_lowp_lowt_5u_no_neck.yaml",
        "exp_cifar10_combined_lowp_lowt_5u_neck.yaml",
        "exp_cifar10_combined_lowp_hight_5u_no_neck.yaml",
        "exp_cifar10_combined_lowp_hight_5u_neck.yaml",
        "exp_cifar10_penalty_highp_nonet_5u_no_neck.yaml",
        "exp_cifar10_penalty_highp_nonet_5u_neck.yaml",
        "exp_cifar10_combined_highp_lowt_5u_no_neck.yaml",
        "exp_cifar10_combined_highp_lowt_5u_neck.yaml",
        "exp_cifar10_combined_highp_hight_5u_no_neck.yaml",
        "exp_cifar10_combined_highp_hight_5u_neck.yaml",
        "exp_gtsrb_threshold_nonep_lowt_3u_no_neck.yaml",
        "exp_gtsrb_threshold_nonep_lowt_3u_neck.yaml",
        "exp_gtsrb_threshold_nonep_hight_3u_no_neck.yaml",
        "exp_gtsrb_threshold_nonep_hight_3u_neck.yaml",
        "exp_gtsrb_penalty_lowp_nonet_3u_no_neck.yaml",
        "exp_gtsrb_penalty_lowp_nonet_3u_neck.yaml",
        "exp_gtsrb_combined_lowp_lowt_3u_no_neck.yaml",
        "exp_gtsrb_combined_lowp_lowt_3u_neck.yaml",
        "exp_gtsrb_combined_lowp_hight_3u_no_neck.yaml",
        "exp_gtsrb_combined_lowp_hight_3u_neck.yaml",
        "exp_gtsrb_penalty_highp_nonet_3u_no_neck.yaml",
        "exp_gtsrb_penalty_highp_nonet_3u_neck.yaml",
        "exp_gtsrb_combined_highp_lowt_3u_no_neck.yaml",
        "exp_gtsrb_combined_highp_lowt_3u_neck.yaml",
        "exp_gtsrb_combined_highp_hight_3u_no_neck.yaml",
        "exp_gtsrb_combined_highp_hight_3u_neck.yaml",
        "exp_gtsrb_threshold_nonep_lowt_6u_no_neck.yaml",
        "exp_gtsrb_threshold_nonep_lowt_6u_neck.yaml",
        "exp_gtsrb_threshold_nonep_hight_6u_no_neck.yaml",
        "exp_gtsrb_threshold_nonep_hight_6u_neck.yaml",
        "exp_gtsrb_penalty_lowp_nonet_6u_no_neck.yaml",
        "exp_gtsrb_penalty_lowp_nonet_6u_neck.yaml",
        "exp_gtsrb_combined_lowp_lowt_6u_no_neck.yaml",
        "exp_gtsrb_combined_lowp_lowt_6u_neck.yaml",
        "exp_gtsrb_combined_lowp_hight_6u_no_neck.yaml",
        "exp_gtsrb_combined_lowp_hight_6u_neck.yaml",
        "exp_gtsrb_penalty_highp_nonet_6u_no_neck.yaml",
        "exp_gtsrb_penalty_highp_nonet_6u_neck.yaml",
        "exp_gtsrb_combined_highp_lowt_6u_no_neck.yaml",
        "exp_gtsrb_combined_highp_lowt_6u_neck.yaml",
        "exp_gtsrb_combined_highp_hight_6u_no_neck.yaml",
        "exp_gtsrb_combined_highp_hight_6u_neck.yaml",
        "exp_gtsrb_threshold_nonep_lowt_9u_no_neck.yaml",
        "exp_gtsrb_threshold_nonep_lowt_9u_neck.yaml",
        "exp_gtsrb_threshold_nonep_hight_9u_no_neck.yaml",
        "exp_gtsrb_threshold_nonep_hight_9u_neck.yaml",
        "exp_gtsrb_penalty_lowp_nonet_9u_no_neck.yaml",
        "exp_gtsrb_penalty_lowp_nonet_9u_neck.yaml",
        "exp_gtsrb_combined_lowp_lowt_9u_no_neck.yaml",
        "exp_gtsrb_combined_lowp_lowt_9u_neck.yaml",
        "exp_gtsrb_combined_lowp_hight_9u_no_neck.yaml",
        "exp_gtsrb_combined_lowp_hight_9u_neck.yaml",
        "exp_gtsrb_penalty_highp_nonet_9u_no_neck.yaml",
        "exp_gtsrb_penalty_highp_nonet_9u_neck.yaml",
        "exp_gtsrb_combined_highp_lowt_9u_no_neck.yaml",
        "exp_gtsrb_combined_highp_lowt_9u_neck.yaml",
        "exp_gtsrb_combined_highp_hight_9u_no_neck.yaml",
        "exp_gtsrb_combined_highp_hight_9u_neck.yaml",
    ]
    
    failed_experiments = []
    
    for i, config_file in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Starting experiment: {config_file}")
        
        if not run_experiment(config_file):
            failed_experiments.append(config_file)
    
    print(f"\n\nSummary:")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {len(experiments) - len(failed_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print("\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")

if __name__ == "__main__":
    main()
