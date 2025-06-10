#!/usr/bin/env python3
"""
Script to train Open Set Recognition models on CIFAR-10 and GTSRB datasets separately
and save results in dataset-named folders.
"""

import subprocess
import sys
import os
import shutil
import json
from pathlib import Path
import yaml

def run_command(cmd, description, cwd=None):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    if cwd:
        print(f"Working directory: {cwd}")
    print(f"{'='*80}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    
    if result.returncode != 0:
        print(f"Error running {description}:")
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        return False, None
    else:
        print("Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True, result.stdout

def get_latest_run_id(runs_dir):
    """Get the latest run ID from the runs directory"""
    if not os.path.exists(runs_dir):
        return None
    
    run_dirs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))])
    return run_dirs[-1] if run_dirs else None

def move_run_to_dataset_folder(run_id, dataset_name, outputs_dir):
    """Move the run from outputs/runs to outputs/{dataset_name}/"""
    source_path = os.path.join(outputs_dir, "runs", run_id)
    target_dir = os.path.join(outputs_dir, dataset_name)
    target_path = os.path.join(target_dir, run_id)
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Move the run folder
    if os.path.exists(source_path):
        shutil.move(source_path, target_path)
        print(f"Moved run {run_id} to {target_path}")
        return target_path
    else:
        print(f"Warning: Source path {source_path} does not exist")
        return None

def main():
    print("Open Set Recognition: Training CIFAR-10 and GTSRB separately")
    print("=" * 80)
    
    # Check if we're in the right directory
    if not os.path.exists("src/deep_osr"):
        print("Error: Please run this script from the root directory of the project")
        sys.exit(1)
    
    # Ensure we have the dl2 conda environment
    conda_env = "dl2"
    
    # Define datasets and their experiment configs
    datasets = [
        {
            "name": "CIFAR-10",
            "folder_name": "cifar10",
            "experiment": "cifar10_resnet50_energy",
            "dataset_config": "cifar10_8k_2u"
        },
        {
            "name": "GTSRB", 
            "folder_name": "gtsrb",
            "experiment": "gtsrb_resnet50_energy",
            "dataset_config": "gtsrb_40k_3u"
        }
    ]
    
    outputs_dir = "outputs"
    results_summary = {}
    
    for dataset in datasets:
        dataset_name = dataset["name"]
        folder_name = dataset["folder_name"] 
        experiment_name = dataset["experiment"]
        dataset_config = dataset["dataset_config"]
        
        print(f"\n{'='*80}")
        print(f"TRAINING {dataset_name.upper()} DATASET")
        print(f"{'='*80}")
        
        # 1. Train the model
        train_cmd = f"conda run -n {conda_env} python -m src.deep_osr.train experiment={experiment_name}"
        success, output = run_command(train_cmd, f"Training {dataset_name} model")
        
        if not success:
            print(f"Failed to train {dataset_name} model. Skipping.")
            results_summary[dataset_name] = {"status": "training_failed"}
            continue
        
        # 2. Get the latest run ID
        latest_run_id = get_latest_run_id(os.path.join(outputs_dir, "runs"))
        if not latest_run_id:
            print(f"No run found after training {dataset_name}. Skipping.")
            results_summary[dataset_name] = {"status": "no_run_found"}
            continue
        
        print(f"Latest {dataset_name} run: {latest_run_id}")
        
        # 3. Move the run to dataset-specific folder
        moved_path = move_run_to_dataset_folder(latest_run_id, folder_name, outputs_dir)
        if not moved_path:
            print(f"Failed to move {dataset_name} run. Continuing with evaluation.")
            moved_path = os.path.join(outputs_dir, "runs", latest_run_id)
        
        # 4. Evaluate the model
        eval_cmd = f"conda run -n {conda_env} python -m src.deep_osr.eval eval.run_id={latest_run_id} dataset={dataset_config}"
        success, eval_output = run_command(eval_cmd, f"Evaluating {dataset_name} model")
        
        if success:
            results_summary[dataset_name] = {
                "status": "completed",
                "run_id": latest_run_id,
                "location": moved_path,
                "experiment": experiment_name,
                "dataset_config": dataset_config
            }
        else:
            results_summary[dataset_name] = {
                "status": "evaluation_failed", 
                "run_id": latest_run_id,
                "location": moved_path,
                "experiment": experiment_name,
                "dataset_config": dataset_config
            }
    
    # 5. Save summary
    summary_file = os.path.join(outputs_dir, "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    
    for dataset_name, info in results_summary.items():
        print(f"\n{dataset_name}:")
        print(f"  Status: {info['status']}")
        if 'run_id' in info:
            print(f"  Run ID: {info['run_id']}")
        if 'location' in info:
            print(f"  Location: {info['location']}")
        if 'experiment' in info:
            print(f"  Experiment: {info['experiment']}")
    
    print(f"\nDetailed summary saved to: {summary_file}")
    
    # 6. Print instructions for accessing results
    print(f"\n{'='*80}")
    print("RESULTS LOCATION")
    print(f"{'='*80}")
    
    for dataset in datasets:
        dataset_name = dataset["name"]
        folder_name = dataset["folder_name"]
        
        if dataset_name in results_summary and results_summary[dataset_name]["status"] in ["completed", "evaluation_failed"]:
            print(f"\n{dataset_name} results:")
            print(f"  Model and logs: outputs/{folder_name}/{results_summary[dataset_name]['run_id']}/")
            print(f"  Evaluation plots: outputs/{folder_name}/{results_summary[dataset_name]['run_id']}/eval_outputs/plots/")
            print(f"  Per-class matrices: outputs/{folder_name}/{results_summary[dataset_name]['run_id']}/eval_outputs/plots/per_class_matrices/")
    
    print(f"\n{'='*80}")
    print("Training and evaluation completed!")
    print("Check the outputs/{dataset_name}/ directories for results and plots.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 