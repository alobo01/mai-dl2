import sys
import os
import pytest
from omegaconf import OmegaConf

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture(scope="session")
def test_data_path():
    """Returns a path for test-specific data, like a small CIFAR10 download."""
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'temp_test_data'))
    os.makedirs(path, exist_ok=True)
    return path

@pytest.fixture
def base_config_dataset(test_data_path):
    """Provides a base OmegaConf DictConfig for dataset configuration for tests."""
    cfg_dict = {
        'name': 'cifar10',
        'path': os.path.join(test_data_path, 'cifar10'), # Test-specific path for CIFAR10
        'num_classes': 10, # Total classes in CIFAR10
        'known_classes': [0, 1, 2], # e.g., airplane, automobile, bird
        'unknown_classes': [8, 9],  # e.g., ship, truck
        'num_known_classes': 3, # Should match len(known_classes)
        'img_size': 32,
        'batch_size': 4,     # Small batch size for testing
        'num_workers': 0,    # Use 0 for easier debugging and to avoid multiprocessing issues in tests
    }
    return OmegaConf.create(cfg_dict)