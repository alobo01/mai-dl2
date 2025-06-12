#!/usr/bin/env python3
"""
Enhanced Open-Set Recognition Experiments Runner
===============================================

This script executes the enhanced OSR experiments with modified NLL loss functions
and dummy class training. It supports parallel execution and comprehensive evaluation.

Usage:
    python run_enhanced_experiments.py [--config-dir configs] [--parallel] [--gpu-ids 0,1,2,3]
"""

import argparse
import multiprocessing
import os
import subprocess
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional
import concurrent.futures
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_experiments.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Enhanced experiment runner with parallel execution support."""
    
    def __init__(self, config_dir: str = "configs", gpu_ids: Optional[List[int]] = None):
        self.config_dir = Path(config_dir)
        self.gpu_ids = gpu_ids if gpu_ids else [0]
        self.results_dir = Path("enhanced_experiment_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Experiment configurations
        self.cifar10_experiments = [
            "exp_cifar10_threshold_1u.yaml",
            "exp_cifar10_penalty_3u.yaml", 
            "exp_cifar10_combined_5u.yaml",
            "exp_cifar10_high_penalty_1u.yaml",
            "exp_cifar10_high_threshold_3u.yaml",
            "exp_cifar10_optimized_combined_5u.yaml"
        ]
        
        self.gtsrb_experiments = [
            "exp_gtsrb_threshold_3u.yaml",
            "exp_gtsrb_penalty_6u.yaml",
            "exp_gtsrb_combined_9u.yaml", 
            "exp_gtsrb_high_penalty_3u.yaml",
            "exp_gtsrb_low_threshold_6u.yaml",
            "exp_gtsrb_optimized_combined_9u.yaml"
        ]
    
    def validate_config(self, config_path: Path) -> bool:
        """Validate experiment configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['defaults', 'experiment', 'dataset', 'model', 'train']
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required section '{section}' in {config_path}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating config {config_path}: {e}")
            return False
    
    def run_single_experiment(self, config_file: str, gpu_id: int) -> dict:
        """Run a single experiment on specified GPU."""
        config_path = self.config_dir / config_file
        
        if not self.validate_config(config_path):
            return {"config": config_file, "status": "failed", "error": "Invalid config"}
        
        experiment_name = config_file.replace(".yaml", "")
        output_dir = self.results_dir / experiment_name
        output_dir.mkdir(exist_ok=True)
        
        # Prepare command
        cmd = [
            sys.executable, "-m", "src.train",
            "--config-path", str(self.config_dir.absolute()),
            "--config-name", config_file.replace(".yaml", ""),
            f"trainer.devices=[{gpu_id}]",
            f"trainer.default_root_dir={output_dir}",
            f"hydra.run.dir={output_dir}/hydra_outputs",
            f"experiment.name={experiment_name}",
            "trainer.enable_progress_bar=False"  # Reduce output noise
        ]
        
        logger.info(f"Starting experiment {experiment_name} on GPU {gpu_id}")
        start_time = time.time()
        
        try:
            # Run experiment
            with open(output_dir / "train.log", "w") as log_file:
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=7200,  # 2 hour timeout
                    cwd=Path.cwd()
                )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"Experiment {experiment_name} completed successfully in {duration:.1f}s")
                
                # Run evaluation
                eval_result = self.run_evaluation(experiment_name, output_dir, gpu_id)
                
                return {
                    "config": config_file,
                    "status": "success", 
                    "duration": duration,
                    "output_dir": str(output_dir),
                    "evaluation": eval_result
                }
            else:
                logger.error(f"Experiment {experiment_name} failed with return code {result.returncode}")
                return {
                    "config": config_file,
                    "status": "failed",
                    "error": f"Process failed with code {result.returncode}",
                    "duration": duration
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"Experiment {experiment_name} timed out")
            return {"config": config_file, "status": "timeout", "duration": 7200}
        except Exception as e:
            logger.error(f"Error running experiment {experiment_name}: {e}")
            return {"config": config_file, "status": "error", "error": str(e)}
    
    def run_evaluation(self, experiment_name: str, output_dir: Path, gpu_id: int) -> dict:
        """Run evaluation for completed experiment."""
        try:
            # Find best checkpoint
            checkpoint_dir = output_dir / "checkpoints"
            if not checkpoint_dir.exists():
                return {"status": "failed", "error": "No checkpoints found"}
            
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if not checkpoints:
                return {"status": "failed", "error": "No checkpoint files found"}
            
            # Use the best checkpoint (assuming filename contains 'best')
            best_checkpoint = None
            for ckpt in checkpoints:
                if "best" in ckpt.name.lower():
                    best_checkpoint = ckpt
                    break
            
            if not best_checkpoint:
                best_checkpoint = checkpoints[-1]  # Use last checkpoint
            
            # Run evaluation script
            eval_cmd = [
                sys.executable, "-m", "src.eval",
                "--checkpoint", str(best_checkpoint),
                "--output-dir", str(output_dir / "evaluation"),
                f"--gpu-id", str(gpu_id)
            ]
            
            eval_result = subprocess.run(
                eval_cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if eval_result.returncode == 0:
                return {"status": "success", "checkpoint": str(best_checkpoint)}
            else:
                return {
                    "status": "failed", 
                    "error": eval_result.stderr,
                    "checkpoint": str(best_checkpoint)
                }
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_experiments_parallel(self, experiment_configs: List[str], max_workers: int = None) -> List[dict]:
        """Run experiments in parallel across available GPUs."""
        if max_workers is None:
            max_workers = min(len(self.gpu_ids), len(experiment_configs))
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit experiments
            future_to_config = {}
            
            for i, config in enumerate(experiment_configs):
                gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
                future = executor.submit(self.run_single_experiment, config, gpu_id)
                future_to_config[future] = config
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed {config}: {result['status']}")
                except Exception as e:
                    logger.error(f"Error with {config}: {e}")
                    results.append({"config": config, "status": "error", "error": str(e)})
        
        return results
    
    def run_experiments_sequential(self, experiment_configs: List[str]) -> List[dict]:
        """Run experiments sequentially on single GPU."""
        results = []
        gpu_id = self.gpu_ids[0]
        
        for config in experiment_configs:
            result = self.run_single_experiment(config, gpu_id)
            results.append(result)
            logger.info(f"Completed {config}: {result['status']}")
            
            # Brief pause between experiments
            time.sleep(5)
        
        return results
    
    def generate_summary_report(self, all_results: List[dict]) -> None:
        """Generate a summary report of all experiments."""
        report_path = self.results_dir / "experiment_summary.txt"
        
        with open(report_path, "w") as f:
            f.write("Enhanced Open-Set Recognition Experiments Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            total_experiments = len(all_results)
            successful = len([r for r in all_results if r["status"] == "success"])
            failed = len([r for r in all_results if r["status"] == "failed"])
            errors = len([r for r in all_results if r["status"] in ["error", "timeout"]])
            
            f.write(f"Total Experiments: {total_experiments}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Errors/Timeouts: {errors}\n\n")
            
            # Individual results
            f.write("Individual Results:\n")
            f.write("-" * 30 + "\n")
            
            for result in all_results:
                f.write(f"Config: {result['config']}\n")
                f.write(f"Status: {result['status']}\n")
                
                if "duration" in result:
                    f.write(f"Duration: {result['duration']:.1f}s\n")
                
                if "error" in result:
                    f.write(f"Error: {result['error']}\n")
                
                if "output_dir" in result:
                    f.write(f"Output: {result['output_dir']}\n")
                
                f.write("\n")
        
        logger.info(f"Summary report saved to {report_path}")
    
    def run_all_experiments(self, parallel: bool = True, dataset: str = "all") -> None:
        """Run all experiments for specified datasets."""
        all_results = []
        
        if dataset in ["all", "cifar10"]:
            logger.info("Running CIFAR-10 experiments...")
            if parallel and len(self.gpu_ids) > 1:
                cifar_results = self.run_experiments_parallel(self.cifar10_experiments)
            else:
                cifar_results = self.run_experiments_sequential(self.cifar10_experiments)
            all_results.extend(cifar_results)
        
        if dataset in ["all", "gtsrb"]:
            logger.info("Running GTSRB experiments...")
            if parallel and len(self.gpu_ids) > 1:
                gtsrb_results = self.run_experiments_parallel(self.gtsrb_experiments)
            else:
                gtsrb_results = self.run_experiments_sequential(self.gtsrb_experiments)
            all_results.extend(gtsrb_results)
        
        # Generate summary
        self.generate_summary_report(all_results)
        
        logger.info("All experiments completed!")
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Run Enhanced OSR Experiments")
    parser.add_argument("--config-dir", default="configs", help="Configuration directory")
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel")
    parser.add_argument("--gpu-ids", default="0", help="Comma-separated GPU IDs (e.g., 0,1,2,3)")
    parser.add_argument("--dataset", choices=["all", "cifar10", "gtsrb"], default="all", 
                       help="Which dataset experiments to run")
    parser.add_argument("--max-workers", type=int, help="Maximum parallel workers")
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    
    # Create runner
    runner = ExperimentRunner(
        config_dir=args.config_dir,
        gpu_ids=gpu_ids
    )
    
    logger.info(f"Starting enhanced OSR experiments on GPUs: {gpu_ids}")
    logger.info(f"Parallel execution: {args.parallel}")
    logger.info(f"Dataset: {args.dataset}")
    
    # Run experiments
    start_time = time.time()
    results = runner.run_all_experiments(
        parallel=args.parallel,
        dataset=args.dataset
    )
    total_time = time.time() - start_time
    
    # Final summary
    successful = len([r for r in results if r["status"] == "success"])
    logger.info(f"Experiments completed in {total_time:.1f}s")
    logger.info(f"Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")


if __name__ == "__main__":
    main()
