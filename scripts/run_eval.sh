#!/usr/bin/env bash

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <RUN_ID_OR_PATH_TO_RUN_DIR>"
  echo "Example: $0 2025-05-16_10-12-00"
  echo "    or $0 outputs/runs/2025-05-16_10-12-00"
  exit 1
fi

RUN_PATH=$1
RUN_ID=$(basename "${RUN_PATH}") # Extract final part of path as RUN_ID

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Running evaluation for run: ${RUN_ID}"
echo "Run directory: ${RUN_PATH}" # This should be the full path to the hydra run dir

# Check if RUN_PATH is a full path or just ID
if [[ ! -d "${RUN_PATH}" ]]; then
    # Assume it's an ID and construct path from outputs_root_dir (defined in config or default "outputs")
    # This requires loading config to get outputs_root_dir, which is complex for a shell script.
    # For simplicity, assume RUN_PATH is the actual path to the run directory.
    echo "Error: Run directory ${RUN_PATH} not found. Please provide the full path to the run directory."
    exit 1
fi

# eval.py expects cfg.eval.run_id to be set to the folder name (e.g., 2025-05-16_10-12-00)
# The cfg.eval.checkpoint_path will be figured out by eval.py if not explicitly set.
# The main config.yaml (and experiment config) for eval will be loaded by Hydra inside eval.py.
# We pass eval.run_id so eval.py knows which training run's config and checkpoints to use.

# The structure of eval.py allows it to load the config from the specified run_id's folder.
# We just need to tell eval.py the run_id.
python -m src.eval \
    eval.run_id="${RUN_ID}" \
    hydra.run.dir="${RUN_PATH}/eval_hydra_logs" \
    "${@:2}" # Pass through additional CLI args if any

# Note: hydra.run.dir for the eval script itself.
# The eval script will then use eval.run_id to find the *training* run's artifacts.
# outputs_root_dir from the *training* config will be used to locate cfg.outputs_root_dir/runs/RUN_ID

echo "Evaluation script finished. Results in ${RUN_PATH}/eval_outputs/"