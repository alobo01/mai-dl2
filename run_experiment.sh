#!/bin/bash

#
# HTCondor job script for running a single experiment
# Usage: ./run_experiment.sh <experiment_config.yaml>
#
set -e

cd /lhome/ext/iiia021/iiia0211/embeddingInspector/osr/ || exit 1

echo "========================================"
echo "Enhanced OSR Experiment Starting"
echo "========================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Experiment Config: $1"
echo "========================================"

# Set job parameters
EXPERIMENT_CONFIG=$1



if [ -z "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: No experiment config provided"
    echo "Usage: $0 <experiment_config.yaml>"
    exit 1
fi



# Set up environment
source /lhome/ext/iiia021/iiia0211/miniconda/etc/profile.d/conda.sh || exit 1
echo "conda activated"
conda env list || exit 1
conda activate osr || exit 1
conda env list || exit 1
which python || exit 1




# Create job-specific output directory
EXPERIMENT_NAME=$(basename "$EXPERIMENT_CONFIG" .yaml)
JOB_OUTPUT_DIR="htcondor_outputs/${EXPERIMENT_NAME}"
mkdir -p "$JOB_OUTPUT_DIR"

echo "Experiment name: $EXPERIMENT_NAME"
echo "Job output directory: $JOB_OUTPUT_DIR"
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
    --config-path "/lhome/ext/iiia021/iiia0211/embeddingInspector/osr/configs/experiments/" \
    --config-name "$EXPERIMENT_CONFIG" 

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
else
    echo "ERROR: Training failed for $EXPERIMENT_NAME with exit code $EXIT_CODE"
fi

echo "Job finished at: $(date)"
exit $EXIT_CODE
