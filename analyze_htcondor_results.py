#!/usr/bin/env python3
"""
HTCondor Results Analysis for Enhanced OSR Experiments
=====================================================

This script analyzes the results downloaded from HTCondor jobs on a slower machine
without GPU requirements. It processes the saved test predictions and generates
comprehensive analysis reports.

Usage:
    python analyze_htcondor_results.py [--results-dir htcondor_outputs] [--output-dir analysis]
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import yaml
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_curve, 
    roc_curve, confusion_matrix, classification_report
)


class HTCondorResultsAnalyzer:
    """Analyzer for HTCondor enhanced OSR experiment results."""
    
    def __init__(self, results_dir: str, output_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.job_results = {}
        self.summary_data = []
        
    def discover_jobs(self):
        """Discover all completed HTCondor jobs."""
        print("Discovering HTCondor job results...")
        
        job_dirs = []
        for item in self.results_dir.iterdir():
            if item.is_dir() and item.name.startswith("job_"):
                job_dirs.append(item)
        
        print(f"Found {len(job_dirs)} job directories")
        return sorted(job_dirs)
    
    def load_job_results(self, job_dir: Path):
        """Load results from a single HTCondor job."""
        job_name = job_dir.name
        print(f"Processing {job_name}...")
        
        results = {
            "job_name": job_name,
            "job_dir": str(job_dir),
            "status": "unknown"
        }
        
        # Extract job info from directory name
        # Format: job_{ID}_{CONFIG_NAME}
        parts = job_name.split("_")
        if len(parts) >= 3:
            results["job_id"] = parts[1]
            results["config_name"] = "_".join(parts[2:])
        
        # Load test predictions CSV
        predictions_file = job_dir / "test_predictions.csv"
        if predictions_file.exists():
            try:
                df_pred = pd.read_csv(predictions_file)
                results["predictions"] = df_pred
                results["num_test_samples"] = len(df_pred)
                results["status"] = "completed"
                print(f"  ✓ Loaded {len(df_pred)} test predictions")
            except Exception as e:
                print(f"  ✗ Error loading predictions: {e}")
                results["status"] = "error"
        else:
            print(f"  ✗ No test predictions found")
            results["status"] = "incomplete"
        
        # Load configuration
        config_file = job_dir / "config_resolved.yaml"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    results["config"] = yaml.safe_load(f)
                print(f"  ✓ Loaded configuration")
            except Exception as e:
                print(f"  ✗ Error loading config: {e}")
        
        # Load metadata
        meta_file = job_dir / "meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    results["meta"] = json.load(f)
                print(f"  ✓ Loaded metadata")
            except Exception as e:
                print(f"  ✗ Error loading metadata: {e}")
        
        # Check for checkpoints
        checkpoints_dir = job_dir / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("*.ckpt"))
            results["num_checkpoints"] = len(checkpoints)
            print(f"  ✓ Found {len(checkpoints)} checkpoints")
        else:
            results["num_checkpoints"] = 0
        
        return results
    
    def compute_metrics_from_predictions(self, df_pred, config):
        """Compute OSR metrics from test predictions."""
        if df_pred is None or len(df_pred) == 0:
            return {}
        
        true_labels = df_pred['true_label'].values
        predicted_labels = df_pred['predicted_label'].values
        osr_scores = df_pred['osr_score'].values
        
        # Get probability columns
        prob_cols = [col for col in df_pred.columns if col.startswith('prob_class_')]
        if prob_cols:
            probabilities = df_pred[prob_cols].values
            max_probs = np.max(probabilities, axis=1)
        else:
            max_probs = np.ones(len(df_pred)) * 0.5
        
        # Determine known vs unknown classes
        num_known_classes = config.get("dataset", {}).get("num_known_classes", 10)
        known_mask = true_labels < num_known_classes
        unknown_mask = ~known_mask
        
        metrics = {}
        
        # Basic accuracy metrics
        if np.any(known_mask):
            known_acc = accuracy_score(true_labels[known_mask], predicted_labels[known_mask])
            metrics["known_accuracy"] = known_acc
        
        # OSR metrics (known vs unknown detection)
        if np.any(unknown_mask) and np.any(known_mask):
            # Binary classification: known (0) vs unknown (1)
            binary_true = (~known_mask).astype(int)
            
            # Use OSR scores if available, otherwise use negative max probability
            if np.std(osr_scores) > 1e-6:
                binary_scores = osr_scores
            else:
                binary_scores = -max_probs
            
            # AUROC
            try:
                auroc = roc_auc_score(binary_true, binary_scores)
                metrics["auroc"] = auroc
            except Exception as e:
                print(f"    Warning: Could not compute AUROC: {e}")
            
            # AUPR
            try:
                precision, recall, _ = precision_recall_curve(binary_true, binary_scores)
                aupr = np.trapz(precision, recall)
                metrics["aupr"] = aupr
            except Exception as e:
                print(f"    Warning: Could not compute AUPR: {e}")
            
            # FPR at 95% TPR
            try:
                fpr, tpr, _ = roc_curve(binary_true, binary_scores)
                tpr_95_idx = np.argmax(tpr >= 0.95)
                if tpr_95_idx < len(fpr):
                    fpr_95 = fpr[tpr_95_idx]
                    metrics["fpr95"] = fpr_95
            except Exception as e:
                print(f"    Warning: Could not compute FPR@95: {e}")
        
        # Overall accuracy
        overall_acc = accuracy_score(true_labels, predicted_labels)
        metrics["overall_accuracy"] = overall_acc
        
        # Class distribution
        metrics["num_known_samples"] = np.sum(known_mask)
        metrics["num_unknown_samples"] = np.sum(unknown_mask)
        metrics["total_samples"] = len(df_pred)
        
        return metrics
    
    def analyze_all_jobs(self):
        """Analyze all discovered HTCondor jobs."""
        job_dirs = self.discover_jobs()
        
        for job_dir in job_dirs:
            results = self.load_job_results(job_dir)
            self.job_results[results["job_name"]] = results
            
            # Compute metrics if predictions are available
            if results["status"] == "completed" and "predictions" in results:
                config = results.get("config", {})
                metrics = self.compute_metrics_from_predictions(results["predictions"], config)
                results["metrics"] = metrics
                
                # Add to summary data
                summary_row = {
                    "job_name": results["job_name"],
                    "job_id": results.get("job_id", "unknown"),
                    "config_name": results.get("config_name", "unknown"),
                    "status": results["status"],
                    "num_test_samples": results.get("num_test_samples", 0),
                    "num_checkpoints": results.get("num_checkpoints", 0)
                }
                summary_row.update(metrics)
                self.summary_data.append(summary_row)
                
                print(f"  ✓ Computed metrics: AUROC={metrics.get('auroc', 'N/A'):.3f}")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\nGenerating summary report...")
        
        df_summary = pd.DataFrame(self.summary_data)
        
        # Save summary CSV
        summary_csv = self.output_dir / "htcondor_summary.csv"
        df_summary.to_csv(summary_csv, index=False)
        print(f"Summary saved to {summary_csv}")
        
        # Generate markdown report
        report_path = self.output_dir / "htcondor_analysis_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# HTCondor Enhanced OSR Experiments Analysis\n\n")
            
            # Overview
            total_jobs = len(self.job_results)
            completed_jobs = len([j for j in self.job_results.values() if j["status"] == "completed"])
            
            f.write("## Overview\n\n")
            f.write(f"- **Total Jobs**: {total_jobs}\n")
            f.write(f"- **Completed Jobs**: {completed_jobs}\n")
            f.write(f"- **Success Rate**: {100 * completed_jobs / total_jobs:.1f}%\n\n")
            
            # Performance Summary
            if not df_summary.empty:
                f.write("## Performance Summary\n\n")
                
                # Best performing jobs
                if "auroc" in df_summary.columns:
                    best_auroc = df_summary.loc[df_summary["auroc"].idxmax()]
                    f.write(f"**Best AUROC**: {best_auroc['auroc']:.4f} ({best_auroc['config_name']})\n\n")
                
                # Average performance by dataset
                cifar_jobs = df_summary[df_summary["config_name"].str.contains("cifar10")]
                gtsrb_jobs = df_summary[df_summary["config_name"].str.contains("gtsrb")]
                
                if not cifar_jobs.empty and "auroc" in cifar_jobs.columns:
                    f.write(f"**CIFAR-10 Average AUROC**: {cifar_jobs['auroc'].mean():.4f}\n")
                
                if not gtsrb_jobs.empty and "auroc" in gtsrb_jobs.columns:
                    f.write(f"**GTSRB Average AUROC**: {gtsrb_jobs['auroc'].mean():.4f}\n\n")
                
                # Detailed results table
                f.write("## Detailed Results\n\n")
                f.write("| Job ID | Config | AUROC | Known Acc | Overall Acc | Samples |\n")
                f.write("|--------|--------|-------|-----------|-------------|----------|\n")
                
                for _, row in df_summary.iterrows():
                    f.write(f"| {row.get('job_id', 'N/A')} | {row.get('config_name', 'N/A')} | "
                           f"{row.get('auroc', 0):.3f} | {row.get('known_accuracy', 0):.3f} | "
                           f"{row.get('overall_accuracy', 0):.3f} | {row.get('num_test_samples', 0)} |\n")
        
        print(f"Analysis report saved to {report_path}")
    
    def generate_visualizations(self):
        """Generate analysis visualizations."""
        print("\nGenerating visualizations...")
        
        if not self.summary_data:
            print("No data available for visualizations")
            return
        
        df_summary = pd.DataFrame(self.summary_data)
        
        # Set up plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('HTCondor Enhanced OSR Results Analysis', fontsize=16)
        
        # Extract dataset and strategy from config names
        df_summary['dataset'] = df_summary['config_name'].apply(
            lambda x: 'CIFAR-10' if 'cifar10' in x else 'GTSRB'
        )
        df_summary['strategy'] = df_summary['config_name'].apply(
            lambda x: 'Threshold' if 'threshold' in x else 
                     ('Penalty' if 'penalty' in x else 'Combined')
        )
        
        # AUROC by strategy
        if 'auroc' in df_summary.columns:
            sns.boxplot(data=df_summary, x='strategy', y='auroc', hue='dataset', ax=axes[0,0])
            axes[0,0].set_title('AUROC by Strategy')
            axes[0,0].set_ylabel('AUROC')
        
        # Known accuracy by strategy
        if 'known_accuracy' in df_summary.columns:
            sns.boxplot(data=df_summary, x='strategy', y='known_accuracy', hue='dataset', ax=axes[0,1])
            axes[0,1].set_title('Known Accuracy by Strategy')
            axes[0,1].set_ylabel('Known Accuracy')
        
        # Overall performance scatter
        if 'auroc' in df_summary.columns and 'known_accuracy' in df_summary.columns:
            sns.scatterplot(data=df_summary, x='known_accuracy', y='auroc', 
                           hue='dataset', style='strategy', s=100, ax=axes[1,0])
            axes[1,0].set_title('AUROC vs Known Accuracy')
            axes[1,0].set_xlabel('Known Accuracy')
            axes[1,0].set_ylabel('AUROC')
        
        # Sample distribution
        if 'num_test_samples' in df_summary.columns:
            sns.barplot(data=df_summary, x='config_name', y='num_test_samples', ax=axes[1,1])
            axes[1,1].set_title('Test Samples per Job')
            axes[1,1].set_ylabel('Number of Test Samples')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        viz_path = self.output_dir / 'htcondor_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {viz_path}")
    
    def run_analysis(self):
        """Run complete HTCondor results analysis."""
        print("Starting HTCondor Enhanced OSR Results Analysis")
        print("=" * 50)
        
        self.analyze_all_jobs()
        self.generate_summary_report()
        self.generate_visualizations()
        
        print(f"\nAnalysis complete! Results saved to {self.output_dir}")
        
        # Print quick summary
        completed_jobs = len([j for j in self.job_results.values() if j["status"] == "completed"])
        total_jobs = len(self.job_results)
        
        print(f"\nQuick Summary:")
        print(f"- Analyzed {total_jobs} HTCondor jobs")
        print(f"- {completed_jobs} completed successfully ({100*completed_jobs/total_jobs:.1f}%)")
        
        if self.summary_data:
            df_summary = pd.DataFrame(self.summary_data)
            if 'auroc' in df_summary.columns:
                print(f"- Best AUROC: {df_summary['auroc'].max():.4f}")
                print(f"- Average AUROC: {df_summary['auroc'].mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze HTCondor Enhanced OSR Results")
    parser.add_argument("--results-dir", default="htcondor_outputs",
                       help="Directory containing HTCondor job outputs")
    parser.add_argument("--output-dir", default="htcondor_analysis",
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not Path(args.results_dir).exists():
        print(f"Error: Results directory '{args.results_dir}' not found")
        print("Please download the HTCondor outputs first")
        sys.exit(1)
    
    analyzer = HTCondorResultsAnalyzer(args.results_dir, args.output_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
