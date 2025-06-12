#!/usr/bin/env python3
"""Simple test to verify seed access"""

import sys
sys.path.append('src')

from omegaconf import OmegaConf

def test_direct_load():
    print("=== Testing direct YAML load ===")
    cfg = OmegaConf.load('configs/experiments/exp_gtsrb_threshold_nonep_lowt_9u_no_neck.yaml')
    print(f"Direct load - seed accessible: {'seed' in cfg}")
    print(f"Direct load - seed value: {cfg.get('seed')}")
    
def test_hydra_compose():
    print("\n=== Testing Hydra compose ===")
    import hydra
    from hydra import initialize, compose
    
    try:
        with initialize(config_path="configs", version_base=None):
            # Test 1: Using base config
            cfg_base = compose(config_name="config")
            print(f"Base config - seed accessible: {'seed' in cfg_base}")
            print(f"Base config - seed value: {cfg_base.get('seed')}")
            
            # Test 2: Using experiment config as main config
            try:
                cfg_exp = compose(config_name="experiments/exp_gtsrb_threshold_nonep_lowt_9u_no_neck")
                print(f"Experiment config - seed accessible: {'seed' in cfg_exp}")
                print(f"Experiment config - seed value: {cfg_exp.get('seed')}")
            except Exception as e:
                print(f"Error with experiment config: {e}")
                
    except Exception as e:
        print(f"Hydra error: {e}")

if __name__ == "__main__":
    test_direct_load()
    test_hydra_compose()
