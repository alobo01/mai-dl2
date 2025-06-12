#!/usr/bin/env python3
"""Test script to reproduce the Hydra seed issue."""

import hydra
from omegaconf import DictConfig
import sys
import os

# Add src to path
sys.path.append('src')

@hydra.main(config_path="configs", config_name="experiments/exp_gtsrb_threshold_nonep_lowt_9u_no_neck", version_base=None)
def main(cfg: DictConfig) -> None:
    print("=== Configuration Debug ===")
    print(f"Config keys: {list(cfg.keys())}")
    print(f"'seed' in cfg: {'seed' in cfg}")
    print(f"cfg.get('seed'): {cfg.get('seed')}")
    print(f"Config structure: {cfg._metadata.struct}")
    
    try:
        seed_value = cfg.seed
        print(f"cfg.seed accessed successfully: {seed_value}")
    except Exception as e:
        print(f"Error accessing cfg.seed: {e}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    main()
