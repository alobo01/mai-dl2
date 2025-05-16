#!/usr/bin/env bash

# Exit on error
set -e

# Default experiment if not provided
EXPERIMENT_CONFIG=${1:-"experiment/cifar10_resnet50_energy"} # Example: experiment=cifar10_resnet50_energy

# Ensure hydra.run.dir is set, or use default from config.yaml
# The config.yaml already sets hydra.run.dir, so this might be redundant unless overriding.
# HYDRA_RUN_DIR_OVERRIDE="hydra.run.dir=outputs/runs/\$(now:%Y-%m-%d_%H-%M-%S)" # Example if needed

# Make sure PYTHONPATH includes src if running scripts from root
export PYTHONPATH=$PYTHONPATH:$(pwd) # Assuming pwd is deep-osr root

echo "Running training with experiment config: ${EXPERIMENT_CONFIG}"

# For single GPU training:
# CUDA_VISIBLE_DEVICES=0 python -m src.train \
#     +${EXPERIMENT_CONFIG} \
#     ${HYDRA_RUN_DIR_OVERRIDE} \
#     "$@" # Pass through any additional CLI args

# Simpler, relying on config.yaml for hydra.run.dir
python -m src.train \
    +${EXPERIMENT_CONFIG} \
    "${@:2}" # Pass through additional CLI args, skipping the first one (experiment config)

echo "Training script finished."
# The actual run directory will be printed by train.py