#!/bin/bash
#
# HTCondor job script for running all Enhanced OSR experiments
# This script sets up the environment and runs a single experiment from the experiments folder
#

echo "========================================"
echo "Enhanced OSR HTCondor Job Starting"
echo "========================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $1"
echo "Experiment Config: $2"
echo "GPU ID: $3"
echo "Max Epochs: $4"
echo "========================================"

# Set job parameters
JOB_ID=$1
EXPERIMENT_CONFIG=$2
GPU_ID=${3:-0}
MAX_EPOCHS=${4:-50}

# Set up environment
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Create job-specific output directory
EXPERIMENT_NAME=$(basename "$EXPERIMENT_CONFIG" .yaml)
JOB_OUTPUT_DIR="htcondor_outputs/job_${JOB_ID}_${EXPERIMENT_NAME}"
mkdir -p "$JOB_OUTPUT_DIR"

echo "Experiment name: $EXPERIMENT_NAME"
echo "Job output directory: $JOB_OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python path: $PYTHONPATH"

# Log system info
echo "GPU Info:"
nvidia-smi || echo "nvidia-smi not available"
echo ""

echo "Python Info:"
python --version
echo ""

echo "PyTorch Info:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" || echo "PyTorch not available"
echo ""

# Verify experiment config exists
FULL_CONFIG_PATH="configs/experiments/$EXPERIMENT_CONFIG"
if [ ! -f "$FULL_CONFIG_PATH" ]; then
    echo "ERROR: Experiment config not found: $FULL_CONFIG_PATH"
    exit 1
fi

echo "Found experiment config: $FULL_CONFIG_PATH"

# Run the training experiment
echo "Starting training with experiment config: experiments/$EXPERIMENT_CONFIG"
echo "Command: python -m src.deep_osr.train --config-name experiments/$EXPERIMENT_CONFIG"

python -m src.deep_osr.train \
    --config-name "experiments/$EXPERIMENT_CONFIG" \
    train.trainer.max_epochs="$MAX_EPOCHS" \
    custom_output_dir="$JOB_OUTPUT_DIR" \
    train.trainer.devices=1 \
    train.trainer.precision=32 \
    dataset.num_workers=4 \
    hydra.job.chdir=False

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "========================================"

# Log final results
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Training completed successfully for $EXPERIMENT_NAME"
    echo "Output directory contents:"
    ls -la "$JOB_OUTPUT_DIR"
    
    # Check if test predictions were saved
    if [ -f "$JOB_OUTPUT_DIR/test_predictions.csv" ]; then
        echo "Test predictions saved successfully"
        wc -l "$JOB_OUTPUT_DIR/test_predictions.csv"
    else
        echo "WARNING: Test predictions not found"
    fi
    
    # Check if checkpoints were saved
    if [ -d "$JOB_OUTPUT_DIR/checkpoints" ]; then
        echo "Checkpoints saved:"
        ls -la "$JOB_OUTPUT_DIR/checkpoints/"
    else
        echo "WARNING: No checkpoints found"
    fi
    
    # Check if results were saved
    if [ -f "$JOB_OUTPUT_DIR/results.json" ]; then
        echo "Results summary:"
        cat "$JOB_OUTPUT_DIR/results.json"
    fi
else
    echo "ERROR: Training failed for $EXPERIMENT_NAME with exit code $EXIT_CODE"
    echo "Last 50 lines of any log files:"
    find "$JOB_OUTPUT_DIR" -name "*.log" -exec tail -50 {} \; 2>/dev/null || echo "No log files found"
fi

echo "Job finished at: $(date)"
exit $EXIT_CODE
