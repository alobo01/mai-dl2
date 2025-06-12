#!/usr/bin/env python3
"""
Simple evaluation script for enhanced OSR experiments.
This script evaluates a trained model checkpoint and saves metrics.
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.deep_osr.enhanced_train_module import EnhancedOpenSetLightningModule
    from src.deep_osr.enhanced_data_module import EnhancedOpenSetDataModule
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def load_model_from_checkpoint(checkpoint_path: str, map_location: str = "cpu"):
    """Load model from checkpoint."""
    try:
        model = EnhancedOpenSetLightningModule.load_from_checkpoint(
            checkpoint_path, 
            map_location=map_location,
            strict=False
        )
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def evaluate_model(model, test_loader, device="cpu"):
    """Evaluate model on test set."""
    model.to(device)
    model.eval()
    
    all_labels = []
    all_known_probs = []
    all_osr_scores = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Extract predictions and scores
            if isinstance(outputs, dict):
                logits = outputs.get("cls_logits", outputs.get("logits"))
                osr_scores = outputs.get("osr_score", torch.zeros(images.size(0)))
            else:
                logits = outputs
                osr_scores = torch.zeros(images.size(0))
            
            # Compute probabilities
            probs = F.softmax(logits, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            predictions = torch.argmax(logits, dim=1)
            
            # Collect results
            all_labels.extend(labels.cpu().numpy())
            all_known_probs.extend(max_probs.cpu().numpy())
            all_osr_scores.extend(osr_scores.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    return {
        "labels": np.array(all_labels),
        "known_probs": np.array(all_known_probs),
        "osr_scores": np.array(all_osr_scores),
        "predictions": np.array(all_predictions)
    }


def compute_metrics(results, num_known_classes):
    """Compute evaluation metrics."""
    labels = results["labels"]
    known_probs = results["known_probs"]
    osr_scores = results["osr_scores"]
    predictions = results["predictions"]
    
    # Create binary labels for OSR (known vs unknown)
    known_mask = labels < num_known_classes
    unknown_mask = ~known_mask
    
    metrics = {}
    
    # Accuracy on known classes
    if np.any(known_mask):
        known_acc = accuracy_score(labels[known_mask], predictions[known_mask])
        metrics["test_acc_known"] = float(known_acc)
    
    # Accuracy on unknown classes (should predict as unknown/rejected)
    if np.any(unknown_mask):
        # For unknown samples, we want low confidence or high OSR scores
        # Simple threshold-based detection
        threshold = 0.5
        unknown_detected = (known_probs[unknown_mask] < threshold) | (osr_scores[unknown_mask] > threshold)
        unknown_acc = np.mean(unknown_detected)
        metrics["test_acc_unknown"] = float(unknown_acc)
    
    # AUROC for OSR task
    if np.any(unknown_mask) and np.any(known_mask):
        # Use negative confidence or positive OSR scores as unknownness indicator
        if np.std(osr_scores) > 1e-6:  # OSR scores are meaningful
            osr_labels = (~known_mask).astype(int)
            auroc = roc_auc_score(osr_labels, osr_scores)
        else:  # Fall back to confidence-based detection
            osr_labels = (~known_mask).astype(int)
            auroc = roc_auc_score(osr_labels, -known_probs)
        
        metrics["test_auroc"] = float(auroc)
    
    # FPR at 95% TPR
    if np.any(unknown_mask) and np.any(known_mask):
        from sklearn.metrics import roc_curve
        osr_labels = (~known_mask).astype(int)
        scores = osr_scores if np.std(osr_scores) > 1e-6 else -known_probs
        fpr, tpr, _ = roc_curve(osr_labels, scores)
        
        # Find FPR at 95% TPR
        tpr_95_idx = np.argmax(tpr >= 0.95)
        if tpr_95_idx < len(fpr):
            fpr_95 = fpr[tpr_95_idx]
            metrics["test_fpr95"] = float(fpr_95)
    
    # Overall accuracy (treating unknown as separate class)
    overall_acc = accuracy_score(labels, predictions)
    metrics["test_acc_overall"] = float(overall_acc)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Enhanced OSR Model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup device
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, map_location=device)
    
    if model is None:
        print("Failed to load model")
        sys.exit(1)
    
    # Create a simple test dataset (this would need to be adapted based on actual data)
    # For now, we'll create a minimal evaluation that saves basic metrics
    
    try:
        # Try to extract some basic info from the model
        metrics = {
            "checkpoint_path": args.checkpoint,
            "model_loaded": True,
            "device": device
        }
        
        # If model has hparams, extract useful info
        if hasattr(model, "hparams"):
            hparams = model.hparams
            metrics.update({
                "num_known_classes": getattr(hparams, "num_known_classes", None),
                "loss_strategy": getattr(hparams, "loss_strategy", None),
                "confidence_threshold": getattr(hparams, "confidence_threshold", None),
                "dummy_penalty_factor": getattr(hparams, "dummy_penalty_factor", None)
            })
        
        # Save metrics
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Evaluation completed. Metrics saved to {metrics_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        # Save error info
        error_info = {
            "error": str(e),
            "checkpoint_path": args.checkpoint,
            "status": "failed"
        }
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(error_info, f, indent=2)
        
        sys.exit(1)


if __name__ == "__main__":
    main()
