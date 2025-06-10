#!/usr/bin/env python3
"""
Script to run Open Set Recognition experiments on both CIFAR-10 and GTSRB datasets
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {description}:")
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        return False
    else:
        print("Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True

def main():
    print("Open Set Recognition Experiments: CIFAR-10 vs GTSRB")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("src/deep_osr"):
        print("Error: Please run this script from the root directory of the project")
        sys.exit(1)
    
    # 1. Train GTSRB model
    train_gtsrb_cmd = "python -m src.deep_osr.train experiment=gtsrb_resnet50_energy"
    if not run_command(train_gtsrb_cmd, "Training GTSRB model"):
        print("Failed to train GTSRB model. Exiting.")
        return
    
    # Get the latest run ID for GTSRB
    runs_dir = "outputs/runs"
    if os.path.exists(runs_dir):
        run_dirs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))])
        if run_dirs:
            latest_gtsrb_run = run_dirs[-1]
            print(f"Latest GTSRB run: {latest_gtsrb_run}")
            
            # 2. Evaluate GTSRB model on GTSRB dataset
            eval_gtsrb_cmd = f"python -m src.deep_osr.eval eval.run_id={latest_gtsrb_run} dataset=gtsrb_40k_3u"
            run_command(eval_gtsrb_cmd, "Evaluating GTSRB model on GTSRB dataset")
            
            # 3. Check if we have a CIFAR-10 trained model
            print("\nLooking for existing CIFAR-10 model...")
            existing_cifar_runs = []
            for run_dir in run_dirs:
                config_path = os.path.join(runs_dir, run_dir, "config_resolved.yaml")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config_content = f.read()
                        if 'cifar10' in config_content and run_dir != latest_gtsrb_run:
                            existing_cifar_runs.append(run_dir)
            
            if existing_cifar_runs:
                latest_cifar_run = existing_cifar_runs[-1]
                print(f"Found existing CIFAR-10 run: {latest_cifar_run}")
                
                # 4. Evaluate CIFAR-10 model on CIFAR-10 dataset
                eval_cifar_cmd = f"python -m src.deep_osr.eval eval.run_id={latest_cifar_run}"
                run_command(eval_cifar_cmd, "Evaluating CIFAR-10 model on CIFAR-10 dataset")
                
                # 5. Cross-evaluation: GTSRB model on CIFAR-10 (for comparison)
                eval_cross1_cmd = f"python -m src.deep_osr.eval eval.run_id={latest_gtsrb_run} dataset=cifar10_8k_2u"
                run_command(eval_cross1_cmd, "Cross-evaluation: GTSRB model on CIFAR-10")
                
                # 6. Cross-evaluation: CIFAR-10 model on GTSRB (for comparison)
                eval_cross2_cmd = f"python -m src.deep_osr.eval eval.run_id={latest_cifar_run} dataset=gtsrb_40k_3u"
                run_command(eval_cross2_cmd, "Cross-evaluation: CIFAR-10 model on GTSRB")
                
            else:
                print("No existing CIFAR-10 model found. Training one...")
                # Train CIFAR-10 model
                train_cifar_cmd = "python -m src.deep_osr.train experiment=cifar10_resnet50_energy"
                if run_command(train_cifar_cmd, "Training CIFAR-10 model"):
                    # Get the new latest run (should be CIFAR-10)
                    new_runs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))])
                    latest_cifar_run = new_runs[-1]
                    
                    # Evaluate the new CIFAR-10 model
                    eval_cifar_cmd = f"python -m src.deep_osr.eval eval.run_id={latest_cifar_run}"
                    run_command(eval_cifar_cmd, "Evaluating CIFAR-10 model on CIFAR-10 dataset")
    
    print("\n" + "="*60)
    print("Experiments completed!")
    print("Check the outputs/runs directory for results and plots.")
    print("="*60)

if __name__ == "__main__":
    main() 