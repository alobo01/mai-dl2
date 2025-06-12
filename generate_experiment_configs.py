#!/usr/bin/env python3
"""
Enhanced OSR Experiment Configuration Generator

This script generates all configurations for comprehensive open-set recognition experiments
with dummy class support, testing various loss strategies, thresholds, penalties, and neck configurations.

Experiments:
- 3 penalty levels (none/low/high) 
- 3 threshold levels (none/low/high)
- 3 unknown class ratios: CIFAR-10 [1,3,5], GTSRB [3,6,9]
- 2 neck configurations (with/without)
- 2 datasets (CIFAR-10, GTSRB)

Total: 3 x 3 x 3 x 2 x 2 = 108 configurations
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any
import argparse

class ConfigGenerator:
    def __init__(self, base_dir: str = "configs"):
        self.base_dir = Path(base_dir)
        self.experiments_dir = self.base_dir / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)
        
        # Configurable parameters
        self.penalty_levels = {
            'none': 1.0,
            'low': 2.0,
            'high': 5.0
        }
        
        self.threshold_levels = {
            'none': 0.0,    # No thresholding
            'low': 0.5,
            'high': 0.8
        }
        
        # Unknown class configurations
        self.unknown_configs = {
            'cifar10': [1, 3, 5],
            'gtsrb': [3, 6, 9]
        }
        
        # Dataset configurations
        self.dataset_configs = {
            'cifar10': {
                'total_classes': 10,
                'img_size': 32,
                'batch_size': 128,
                'backbone_features': 2048  # ResNet50
            },
            'gtsrb': {
                'total_classes': 43,
                'img_size': 32,
                'batch_size': 128,
                'backbone_features': 2048  # ResNet50
            }
        }
    
    def set_penalty_levels(self, none: float = 1.0, low: float = 2.0, high: float = 5.0):
        """Configure penalty levels"""
        self.penalty_levels = {'none': none, 'low': low, 'high': high}
    
    def set_threshold_levels(self, none: float = 0.0, low: float = 0.5, high: float = 0.8):
        """Configure threshold levels"""
        self.threshold_levels = {'none': none, 'low': low, 'high': high}
    
    def generate_dataset_config(self, dataset: str, num_unknown: int) -> Dict[str, Any]:
        """Generate dataset configuration"""
        config = self.dataset_configs[dataset]
        num_known = config['total_classes'] - num_unknown
        
        known_ids = list(range(num_known))
        unknown_ids = list(range(num_known, config['total_classes']))
        
        return {
            'name': dataset,
            'path': './data',
            'img_size': config['img_size'],
            'num_classes': num_known,
            'num_known_classes': num_known,
            'total_original_classes': config['total_classes'],
            'known_classes_original_ids': known_ids,
            'unknown_classes_original_ids': unknown_ids,
            'batch_size': config['batch_size'],
            'num_workers': 4,
            'val_split_ratio': 0.2,
            'enhanced_settings': {
                'include_unknown_in_training': True,
                'unknown_training_ratio': 0.3,
                'dummy_class_enabled': True
            }
        }
    
    def generate_model_config(self, dataset: str, use_neck: bool) -> Dict[str, Any]:
        """Generate model configuration"""
        backbone_features = self.dataset_configs[dataset]['backbone_features']
        d_embed = 512 if use_neck else backbone_features
        
        config = {
            'type': "enhanced_openset_resnet50",
            'd_embed': d_embed,
            'backbone': {
                'name': "resnet50",
                'pretrained': True,
                'frozen': False,
                'num_output_features': backbone_features
            },
            'neck': {
                'enabled': use_neck,
                'in_features': backbone_features,
                'out_features': 512,
                'use_batchnorm': True,
                'use_relu': True
            } if use_neck else {
                'enabled': False
            },
            'cls_head': {
                'in_features': d_embed,
                'use_weight_norm': True,
                'label_smoothing': 0.1,
                'temperature': 1.0
            },
            'osr_head': {
                'type': "energy",
                'in_features': d_embed
            }
        }
        
        return config
    
    def generate_train_config(self, strategy: str, penalty_level: str, threshold_level: str) -> Dict[str, Any]:
        """Generate training configuration"""
        penalty_value = self.penalty_levels[penalty_level]
        threshold_value = self.threshold_levels[threshold_level]
        
        # Determine loss strategy based on parameters
        if strategy == "combined":
            loss_strategy = "combined"
        elif penalty_level != 'none' and threshold_level == 'none':
            loss_strategy = "penalty"
        elif threshold_level != 'none' and penalty_level == 'none':
            loss_strategy = "threshold"
        elif penalty_level != 'none' and threshold_level != 'none':
            loss_strategy = "combined"
        else:
            loss_strategy = "standard"
        
        config = {
            'optimizer': {
                'name': "AdamW",
                'lr': 0.001,
                'weight_decay': 0.01
            },
            'scheduler': {
                'name': "CosineAnnealingLR",
                'params': {
                    'T_max': 50
                }
            },
            'trainer': {
                'max_epochs': 50,
                'gpus': 1,
                'precision': 32,
                'deterministic': True,
                'enable_model_summary': True,
                'enable_checkpointing': True,
                'log_every_n_steps': 50,
                'enable_progress_bar': True,
                'logger': True
            },
            'loss': {
                'strategy': loss_strategy,
                'confidence_threshold': threshold_value,
                'dummy_class_penalty': penalty_value,
                'ce_seen_weight': 1.0,
                'dummy_loss_weight': 1.0,
                'osr_loss_weight': 0.1,
                'use_dummy_class': True,  # Always use dummy class
                'threshold_weight': 0.5,
                'penalty_weight': 0.5
            },
            'calibration': {
                'method': "temperature_scaling"
            },
            'run_test_after_train': True,
            'run_custom_evaluation_after_train': True,
            
        }
        
        return config
    
    def generate_experiment_config(self, dataset: str, num_unknown: int, penalty_level: str, 
                                 threshold_level: str, use_neck: bool) -> Dict[str, Any]:
        """Generate complete experiment configuration"""
        
        # Create experiment name
        neck_suffix = "neck" if use_neck else "no_neck"
        strategy = self._get_strategy_name(penalty_level, threshold_level)
        exp_name = f"{dataset}_{strategy}_{penalty_level}p_{threshold_level}t_{num_unknown}u_{neck_suffix}"
        
        config = {
            'experiment': {
                'name': exp_name,
                'use_enhanced': True
            },
            'dataset': self.generate_dataset_config(dataset, num_unknown),
            'model': self.generate_model_config(dataset, use_neck),
            'train': self.generate_train_config(strategy, penalty_level, threshold_level),
            'seed': 42,
            'outputs_root_dir': "outputs",
            'data_root_dir': "data/processed",
            'experiment_name': exp_name
        }
        
        return config
    
    def _get_strategy_name(self, penalty_level: str, threshold_level: str) -> str:
        """Get strategy name based on penalty and threshold levels"""
        if penalty_level != 'none' and threshold_level != 'none':
            return "combined"
        elif penalty_level != 'none':
            return "penalty"
        elif threshold_level != 'none':
            return "threshold"
        else:
            return "standard"
    
    def generate_all_configs(self):
        """Generate all experiment configurations"""
        configs_generated = []
        
        for dataset in ['cifar10', 'gtsrb']:
            for num_unknown in self.unknown_configs[dataset]:
                for penalty_level in self.penalty_levels.keys():
                    for threshold_level in self.threshold_levels.keys():
                        for use_neck in [False, True]:
                            
                            # Skip invalid combinations
                            if penalty_level == 'none' and threshold_level == 'none':
                                continue  # Standard strategy not interesting for this study
                            
                            config = self.generate_experiment_config(
                                dataset, num_unknown, penalty_level, threshold_level, use_neck
                            )
                            
                            filename = f"exp_{config['experiment']['name']}.yaml"
                            filepath = self.experiments_dir / filename
                            
                            with open(filepath, 'w') as f:
                                yaml.dump(config, f, default_flow_style=False, indent=2)
                            
                            configs_generated.append({
                                'filename': filename,
                                'experiment_name': config['experiment']['name'],
                                'dataset': dataset,
                                'num_unknown': num_unknown,
                                'penalty_level': penalty_level,
                                'threshold_level': threshold_level,
                                'use_neck': use_neck,
                                'strategy': self._get_strategy_name(penalty_level, threshold_level)
                            })
        
        return configs_generated
    
    def generate_summary_file(self, configs_generated: List[Dict]):
        """Generate summary file of all experiments"""
        summary_path = self.experiments_dir / "experiments_summary.yaml"
        
        summary = {
            'total_experiments': len(configs_generated),
            'parameters': {
                'penalty_levels': self.penalty_levels,
                'threshold_levels': self.threshold_levels,
                'unknown_configs': self.unknown_configs,
                'datasets': list(self.dataset_configs.keys())
            },
            'experiments': configs_generated
        }
        
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
        
        print(f"Generated {len(configs_generated)} experiment configurations")
        print(f"Summary saved to: {summary_path}")
        
        # Print breakdown
        by_dataset = {}
        by_strategy = {}
        
        for config in configs_generated:
            dataset = config['dataset']
            strategy = config['strategy']
            
            by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
            by_strategy[strategy] = by_strategy.get(strategy, 0) + 1
        
        print("\nBreakdown by dataset:")
        for dataset, count in by_dataset.items():
            print(f"  {dataset}: {count} experiments")
        
        print("\nBreakdown by strategy:")
        for strategy, count in by_strategy.items():
            print(f"  {strategy}: {count} experiments")
    
    def generate_run_scripts(self, configs_generated: List[Dict]):
        """Generate convenience scripts to run experiments"""
        
        # Generate individual run script
        run_script_path = self.base_dir.parent / "run_all_experiments.py"
        run_script_content = '''#!/usr/bin/env python3
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
'''
        
        for config in configs_generated:
            run_script_content += f'        "{config["filename"]}",\n'
        
        run_script_content += '''    ]
    
    failed_experiments = []
    
    for i, config_file in enumerate(experiments, 1):
        print(f"\\n[{i}/{len(experiments)}] Starting experiment: {config_file}")
        
        if not run_experiment(config_file):
            failed_experiments.append(config_file)
    
    print(f"\\n\\nSummary:")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {len(experiments) - len(failed_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        print("\\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")

if __name__ == "__main__":
    main()
'''
        
        with open(run_script_path, 'w', encoding='utf-8') as f:
            f.write(run_script_content)
        
        # Make script executable on Unix systems
        try:
            os.chmod(run_script_path, 0o755)
        except:
            pass
        
        print(f"Generated run script: {run_script_path}")
        
        # Generate batch script for Windows
        batch_script_path = self.base_dir.parent / "run_all_experiments.bat"
        batch_content = "@echo off\n"
        for config in configs_generated:
            batch_content += f'python train_enhanced_osr.py --config configs/experiments/{config["filename"]}\n'
        
        with open(batch_script_path, 'w', encoding='utf-8') as f:
            f.write(batch_content)
        
        print(f"Generated batch script: {batch_script_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Enhanced OSR Experiment Configurations")
    
    parser.add_argument("--penalty-none", type=float, default=1.0, 
                       help="Penalty level for 'none' (default: 1.0)")
    parser.add_argument("--penalty-low", type=float, default=2.0,
                       help="Penalty level for 'low' (default: 2.0)")
    parser.add_argument("--penalty-high", type=float, default=5.0,
                       help="Penalty level for 'high' (default: 5.0)")
    
    parser.add_argument("--threshold-none", type=float, default=0.0,
                       help="Threshold level for 'none' (default: 0.0)")
    parser.add_argument("--threshold-low", type=float, default=0.5,
                       help="Threshold level for 'low' (default: 0.5)")
    parser.add_argument("--threshold-high", type=float, default=0.8,
                       help="Threshold level for 'high' (default: 0.8)")
    
    parser.add_argument("--configs-dir", type=str, default="configs",
                       help="Directory to store generated configs (default: configs)")
    
    args = parser.parse_args()
    
    # Create generator
    generator = ConfigGenerator(args.configs_dir)
    
    # Set configurable parameters
    generator.set_penalty_levels(args.penalty_none, args.penalty_low, args.penalty_high)
    generator.set_threshold_levels(args.threshold_none, args.threshold_low, args.threshold_high)
    
    print("Enhanced OSR Experiment Configuration Generator")
    print("=" * 50)
    print(f"Penalty levels: {generator.penalty_levels}")
    print(f"Threshold levels: {generator.threshold_levels}")
    print(f"Unknown class configs: {generator.unknown_configs}")
    print("=" * 50)
    
    # Generate all configurations
    configs_generated = generator.generate_all_configs()
    
    # Generate summary and run scripts
    generator.generate_summary_file(configs_generated)
    generator.generate_run_scripts(configs_generated)
    
    print("\n✓ All configurations generated successfully!")
    print(f"Configs saved to: {generator.experiments_dir}")


if __name__ == "__main__":
    main()
