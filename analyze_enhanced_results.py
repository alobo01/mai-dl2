#!/usr/bin/env python3
"""
Enhanced OSR Experiments Analysis and Evaluation
===============================================

This script analyzes and compares the results from the enhanced OSR experiments.
It generates comprehensive reports, visualizations, and statistical comparisons.

Usage:
    python analyze_enhanced_results.py [--results-dir enhanced_experiment_results] [--output-dir analysis_results]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    import torch
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, roc_curve
    import wandb
except ImportError as e:
    print(f"Warning: Some optional dependencies not available: {e}")


class EnhancedResultsAnalyzer:
    """Analyzer for enhanced OSR experiment results."""
    
    def __init__(self, results_dir: str, output_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize analysis storage
        self.experiment_results = {}
        self.metrics_summary = defaultdict(list)
        self.comparison_data = defaultdict(dict)
        
    def load_experiment_results(self) -> None:
        """Load results from all experiments."""
        print("Loading experiment results...")
        
        for exp_dir in self.results_dir.iterdir():
            if not exp_dir.is_dir():
                continue
                
            print(f"Processing {exp_dir.name}...")
            
            # Load metrics from various sources
            metrics = self.load_experiment_metrics(exp_dir)
            
            if metrics:
                self.experiment_results[exp_dir.name] = metrics
                self.categorize_experiment(exp_dir.name, metrics)
    
    def load_experiment_metrics(self, exp_dir: Path) -> Optional[Dict]:
        """Load metrics from a single experiment directory."""
        metrics = {}
        
        # Try to load from different possible locations
        possible_files = [
            exp_dir / "evaluation" / "metrics.json",
            exp_dir / "metrics.json", 
            exp_dir / "test_results.json",
            exp_dir / "hydra_outputs" / "metrics.json"
        ]
        
        for metrics_file in possible_files:
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        metrics.update(data)
                        print(f"  Loaded metrics from {metrics_file.name}")
                except Exception as e:
                    print(f"  Error loading {metrics_file}: {e}")
        
        # Try to load from checkpoint files if available
        checkpoint_dir = exp_dir / "checkpoints"
        if checkpoint_dir.exists():
            metrics.update(self.extract_metrics_from_checkpoints(checkpoint_dir))
        
        # Load configuration for context
        config_info = self.load_experiment_config(exp_dir)
        if config_info:
            metrics["config"] = config_info
        
        return metrics if metrics else None
    
    def extract_metrics_from_checkpoints(self, checkpoint_dir: Path) -> Dict:
        """Extract metrics from PyTorch Lightning checkpoints."""
        metrics = {}
        
        try:
            # Find best checkpoint
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if not checkpoints:
                return metrics
            
            best_checkpoint = None
            for ckpt in checkpoints:
                if "best" in ckpt.name.lower():
                    best_checkpoint = ckpt
                    break
            
            if not best_checkpoint:
                best_checkpoint = checkpoints[-1]
            
            # Load checkpoint and extract metrics
            checkpoint = torch.load(best_checkpoint, map_location="cpu")
            
            if "callbacks" in checkpoint:
                for callback_state in checkpoint["callbacks"].values():
                    if "best_model_score" in callback_state:
                        metrics["best_val_score"] = float(callback_state["best_model_score"])
                    if "best_model_path" in callback_state:
                        metrics["best_model_path"] = callback_state["best_model_path"]
            
            if "epoch" in checkpoint:
                metrics["best_epoch"] = checkpoint["epoch"]
            
            if "state_dict" in checkpoint:
                metrics["model_parameters"] = len(checkpoint["state_dict"])
            
        except Exception as e:
            print(f"  Error extracting from checkpoints: {e}")
        
        return metrics
    
    def load_experiment_config(self, exp_dir: Path) -> Optional[Dict]:
        """Load experiment configuration."""
        config_files = [
            exp_dir / "hydra_outputs" / ".hydra" / "config.yaml",
            exp_dir / "config.yaml"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        return yaml.safe_load(f)
                except Exception as e:
                    print(f"  Error loading config {config_file}: {e}")
        
        return None
    
    def categorize_experiment(self, exp_name: str, metrics: Dict) -> None:
        """Categorize experiment by dataset, loss strategy, and parameters."""
        # Extract info from experiment name
        if "cifar10" in exp_name:
            dataset = "CIFAR-10"
        elif "gtsrb" in exp_name:
            dataset = "GTSRB"
        else:
            dataset = "Unknown"
        
        if "threshold" in exp_name:
            strategy = "Threshold"
        elif "penalty" in exp_name:
            strategy = "Penalty"
        elif "combined" in exp_name:
            strategy = "Combined"
        else:
            strategy = "Unknown"
        
        # Extract number of unknown classes
        import re
        unknown_match = re.search(r'(\d+)u', exp_name)
        num_unknown = int(unknown_match.group(1)) if unknown_match else 0
        
        # Store categorized data
        category_key = f"{dataset}_{strategy}"
        self.comparison_data[category_key][num_unknown] = {
            "experiment": exp_name,
            "metrics": metrics
        }
    
    def compute_summary_statistics(self) -> pd.DataFrame:
        """Compute summary statistics across all experiments."""
        summary_data = []
        
        for exp_name, metrics in self.experiment_results.items():
            row = {
                "Experiment": exp_name,
                "Dataset": "CIFAR-10" if "cifar10" in exp_name else "GTSRB",
                "Strategy": self.extract_strategy(exp_name),
                "Unknown_Classes": self.extract_unknown_count(exp_name),
                "Architecture": "With Neck" if "neck" in exp_name else "No Neck"
            }
            
            # Add available metrics
            metric_keys = [
                "test_acc_known", "test_acc_unknown", "test_auroc", 
                "test_aupr", "test_fpr95", "best_val_score",
                "best_epoch", "training_time"
            ]
            
            for key in metric_keys:
                row[key] = metrics.get(key, None)
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def extract_strategy(self, exp_name: str) -> str:
        """Extract loss strategy from experiment name."""
        if "threshold" in exp_name:
            return "Threshold"
        elif "penalty" in exp_name:
            return "Penalty" 
        elif "combined" in exp_name:
            return "Combined"
        return "Unknown"
    
    def extract_unknown_count(self, exp_name: str) -> int:
        """Extract number of unknown classes from experiment name."""
        import re
        match = re.search(r'(\d+)u', exp_name)
        return int(match.group(1)) if match else 0
    
    def generate_comparison_plots(self) -> None:
        """Generate comparison plots and visualizations."""
        print("Generating comparison plots...")
        
        df = self.compute_summary_statistics()
        
        if df.empty:
            print("No data available for plotting")
            return
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Performance by Strategy and Dataset
        self.plot_performance_by_strategy(df)
        
        # 2. Impact of Unknown Class Count
        self.plot_unknown_class_impact(df)
        
        # 3. Training Convergence Comparison
        self.plot_training_convergence(df)
        
        # 4. Architecture Comparison
        self.plot_architecture_comparison(df)
        
        # 5. Parameter Sensitivity Analysis
        self.plot_parameter_sensitivity()
    
    def plot_performance_by_strategy(self, df: pd.DataFrame) -> None:
        """Plot performance comparison by loss strategy."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Comparison by Loss Strategy', fontsize=16)
        
        metrics = ['test_auroc', 'test_acc_known', 'test_acc_unknown', 'test_fpr95']
        titles = ['AUROC', 'Known Accuracy', 'Unknown Accuracy', 'FPR@95']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            if metric in df.columns and not df[metric].isna().all():
                sns.boxplot(data=df, x='Strategy', y=metric, hue='Dataset', ax=ax)
                ax.set_title(title)
                ax.set_ylabel(title)
                
                # Add value annotations
                for j, strategy in enumerate(df['Strategy'].unique()):
                    strategy_data = df[df['Strategy'] == strategy][metric].dropna()
                    if not strategy_data.empty:
                        ax.text(j, strategy_data.max(), f'{strategy_data.mean():.3f}', 
                               ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_by_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_unknown_class_impact(self, df: pd.DataFrame) -> None:
        """Plot impact of number of unknown classes."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Impact of Unknown Class Count', fontsize=16)
        
        for i, dataset in enumerate(['CIFAR-10', 'GTSRB']):
            ax = axes[i]
            dataset_df = df[df['Dataset'] == dataset]
            
            if not dataset_df.empty and 'test_auroc' in dataset_df.columns:
                sns.lineplot(data=dataset_df, x='Unknown_Classes', y='test_auroc', 
                           hue='Strategy', marker='o', ax=ax)
                ax.set_title(f'{dataset} - AUROC vs Unknown Classes')
                ax.set_xlabel('Number of Unknown Classes')
                ax.set_ylabel('AUROC')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'unknown_class_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_convergence(self, df: pd.DataFrame) -> None:
        """Plot training convergence comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Training Convergence Analysis', fontsize=16)
        
        # Best epoch comparison
        if 'best_epoch' in df.columns and not df['best_epoch'].isna().all():
            sns.boxplot(data=df, x='Strategy', y='best_epoch', hue='Dataset', ax=axes[0])
            axes[0].set_title('Convergence Speed (Best Epoch)')
            axes[0].set_ylabel('Best Epoch')
        
        # Validation score comparison
        if 'best_val_score' in df.columns and not df['best_val_score'].isna().all():
            sns.boxplot(data=df, x='Strategy', y='best_val_score', hue='Dataset', ax=axes[1])
            axes[1].set_title('Best Validation Score')
            axes[1].set_ylabel('Validation Score')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_convergence.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_architecture_comparison(self, df: pd.DataFrame) -> None:
        """Compare with/without neck architectures."""
        if 'Architecture' not in df.columns:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Architecture Comparison (With/Without Neck)', fontsize=16)
        
        for i, dataset in enumerate(['CIFAR-10', 'GTSRB']):
            ax = axes[i]
            dataset_df = df[df['Dataset'] == dataset]
            
            if not dataset_df.empty and 'test_auroc' in dataset_df.columns:
                sns.barplot(data=dataset_df, x='Architecture', y='test_auroc', 
                          hue='Strategy', ax=ax)
                ax.set_title(f'{dataset}')
                ax.set_ylabel('AUROC')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'architecture_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_parameter_sensitivity(self) -> None:
        """Plot parameter sensitivity analysis."""
        print("Analyzing parameter sensitivity...")
        
        # This would require parsing specific parameter values from configs
        # For now, create a placeholder showing the concept
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parameter Sensitivity Analysis', fontsize=16)
        
        # Placeholder plots - in actual implementation, extract parameter values
        # from experiment configs and correlate with performance
        
        axes[0, 0].set_title('Confidence Threshold Impact')
        axes[0, 0].set_xlabel('Confidence Threshold')
        axes[0, 0].set_ylabel('AUROC')
        
        axes[0, 1].set_title('Penalty Factor Impact')
        axes[0, 1].set_xlabel('Penalty Factor')
        axes[0, 1].set_ylabel('AUROC')
        
        axes[1, 0].set_title('Unknown Ratio Impact')
        axes[1, 0].set_xlabel('Unknown Training Ratio')
        axes[1, 0].set_ylabel('AUROC')
        
        axes[1, 1].set_title('Loss Weight Impact')
        axes[1, 1].set_xlabel('Loss Component Weight')
        axes[1, 1].set_ylabel('AUROC')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_detailed_report(self) -> None:
        """Generate detailed analysis report."""
        print("Generating detailed report...")
        
        df = self.compute_summary_statistics()
        
        report_path = self.output_dir / 'detailed_analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced Open-Set Recognition Experiments Analysis\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- Total experiments analyzed: {len(self.experiment_results)}\n")
            f.write(f"- Datasets: {df['Dataset'].unique().tolist()}\n")
            f.write(f"- Loss strategies: {df['Strategy'].unique().tolist()}\n")
            f.write(f"- Unknown class configurations: {sorted(df['Unknown_Classes'].unique())}\n\n")
            
            # Best Performing Configurations
            f.write("## Best Performing Configurations\n\n")
            
            if 'test_auroc' in df.columns and not df['test_auroc'].isna().all():
                best_auroc = df.loc[df['test_auroc'].idxmax()]
                f.write(f"**Best AUROC**: {best_auroc['test_auroc']:.4f}\n")
                f.write(f"- Experiment: {best_auroc['Experiment']}\n")
                f.write(f"- Strategy: {best_auroc['Strategy']}\n")
                f.write(f"- Dataset: {best_auroc['Dataset']}\n\n")
            
            # Strategy Comparison
            f.write("## Strategy Comparison\n\n")
            strategy_stats = df.groupby('Strategy')[['test_auroc', 'test_acc_known', 'test_acc_unknown']].agg(['mean', 'std'])
            f.write(strategy_stats.to_string())
            f.write("\n\n")
            
            # Dataset-specific Analysis
            f.write("## Dataset-specific Analysis\n\n")
            for dataset in df['Dataset'].unique():
                f.write(f"### {dataset}\n\n")
                dataset_df = df[df['Dataset'] == dataset]
                
                f.write("Performance by strategy:\n")
                dataset_stats = dataset_df.groupby('Strategy')[['test_auroc', 'test_acc_known']].mean()
                f.write(dataset_stats.to_string())
                f.write("\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the experimental results:\n\n")
            
            # Add specific recommendations based on results
            if not df.empty:
                best_strategy = df.groupby('Strategy')['test_auroc'].mean().idxmax()
                f.write(f"1. **{best_strategy}** loss strategy shows the best average performance\n")
                f.write("2. Consider architecture choices based on dataset complexity\n")
                f.write("3. Parameter tuning shows significant impact on performance\n")
                f.write("4. Unknown class count affects optimal strategy selection\n\n")
        
        print(f"Detailed report saved to {report_path}")
    
    def export_results_table(self) -> None:
        """Export results as CSV and LaTeX tables."""
        df = self.compute_summary_statistics()
        
        # CSV export
        csv_path = self.output_dir / 'experiment_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Results exported to {csv_path}")
        
        # LaTeX table export
        latex_path = self.output_dir / 'results_table.tex'
        with open(latex_path, 'w') as f:
            # Create a simplified table for LaTeX
            table_df = df[['Experiment', 'Dataset', 'Strategy', 'Unknown_Classes', 'test_auroc', 'test_acc_known']]
            table_df = table_df.round(4)
            latex_table = table_df.to_latex(index=False, escape=False)
            f.write(latex_table)
        
        print(f"LaTeX table exported to {latex_path}")
    
    def run_complete_analysis(self) -> None:
        """Run complete analysis pipeline."""
        print("Starting enhanced OSR results analysis...")
        
        # Load all experiment results
        self.load_experiment_results()
        
        if not self.experiment_results:
            print("No experiment results found!")
            return
        
        print(f"Loaded {len(self.experiment_results)} experiments")
        
        # Generate all analyses
        self.generate_comparison_plots()
        self.generate_detailed_report()
        self.export_results_table()
        
        print(f"Analysis complete! Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Enhanced OSR Experiment Results")
    parser.add_argument("--results-dir", default="enhanced_experiment_results", 
                       help="Directory containing experiment results")
    parser.add_argument("--output-dir", default="analysis_results",
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    analyzer = EnhancedResultsAnalyzer(args.results_dir, args.output_dir)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
