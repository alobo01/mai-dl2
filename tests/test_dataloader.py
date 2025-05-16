import pytest
import torch
from deep_osr.data.dataset import OpenSetDataModule
from omegaconf import OmegaConf # Already in conftest, but good to have explicit here

# Uses base_config_dataset fixture from conftest.py

def test_openset_datamodule_setup(base_config_dataset):
    """Test the setup method of OpenSetDataModule."""
    dm = OpenSetDataModule(base_config_dataset)
    dm.prepare_data()  # Downloads CIFAR10 if not present to the test path
    dm.setup(stage='fit') # For train_dataloader
    dm.setup(stage='validate') # For val_dataloader
    dm.setup(stage='test') # For test_dataloader related (test_known, test_unknown)

    assert dm.train_dataset_full is not None, "Training dataset should be initialized"
    assert dm.val_dataset_full is not None, "Validation dataset should be initialized"
    assert dm.test_dataset_known is not None, "Test known dataset should be initialized"
    assert dm.test_dataset_unknown is not None, "Test unknown dataset should be initialized"

    # Check if known_classes and unknown_classes are correctly assigned
    # This requires inspecting the datasets themselves, which is a bit deep.
    # For now, trust that OpenSetDataset filters correctly.
    # A more thorough test would iterate through some samples.
    assert len(dm.train_dataset_full) > 0
    assert len(dm.val_dataset_full) > 0
    # CIFAR10 test has 1000 images per class.
    # Known: 3 classes * 1000 = 3000 total in original test set.
    # Unknown: 2 classes * 1000 = 2000 total in original test set.
    # The actual numbers depend on torchvision's CIFAR10 structure and Subset logic.
    # For CIFAR10, test_base has 10000 samples.
    # If classes [0,1,2] are known, test_dataset_known should have ~3000 samples.
    # If classes [8,9] are unknown, test_dataset_unknown should have ~2000 samples.
    # These are approximate due to Subset behavior.
    # A simple check:
    num_original_test_samples_per_class = 1000 # For CIFAR10
    expected_known_test_size = len(base_config_dataset.known_classes) * num_original_test_samples_per_class
    expected_unknown_test_size = len(base_config_dataset.unknown_classes) * num_original_test_samples_per_class
    
    assert len(dm.test_dataset_known) == expected_known_test_size
    assert len(dm.test_dataset_unknown) == expected_unknown_test_size


def test_train_dataloader_batch(base_config_dataset):
    """Test a batch from the training dataloader."""
    dm = OpenSetDataModule(base_config_dataset)
    dm.prepare_data()
    dm.setup(stage='fit')
    
    loader = dm.train_dataloader()
    images, labels_known_idx, is_known_target = next(iter(loader))

    K = len(base_config_dataset.known_classes)

    assert images.shape == (base_config_dataset.batch_size, 3, 32, 32)
    assert labels_known_idx.shape == (base_config_dataset.batch_size,)
    assert is_known_target.shape == (base_config_dataset.batch_size,)
    
    assert torch.all(is_known_target), "All samples in training set should be known"
    assert torch.all(labels_known_idx >= 0) and torch.all(labels_known_idx < K), \
        f"Training labels should be in range [0, {K-1}]"

def test_val_dataloader_batch(base_config_dataset):
    """Test a batch from the validation dataloader for mixed known/unknown samples."""
    dm = OpenSetDataModule(base_config_dataset)
    dm.prepare_data()
    dm.setup(stage='validate')

    loader = dm.val_dataloader()
    # Validation set can be small or not perfectly balanced, so iterate a few times
    # to increase chance of seeing both known and unknown if present.
    
    seen_known = False
    seen_unknown = False
    K = len(base_config_dataset.known_classes)

    for _ in range(min(5, len(loader))): # Check a few batches
        images, labels_known_idx, is_known_target = next(iter(loader))
        
        assert images.shape[0] <= base_config_dataset.batch_size # Last batch can be smaller
        assert images.shape[1:] == (3, 32, 32)
        assert labels_known_idx.shape[0] == images.shape[0]
        assert is_known_target.shape[0] == images.shape[0]

        if torch.any(is_known_target):
            seen_known = True
            # For known samples, labels must be in [0, K-1]
            assert torch.all(labels_known_idx[is_known_target] >= 0) and \
                   torch.all(labels_known_idx[is_known_target] < K)
        
        if torch.any(~is_known_target):
            seen_unknown = True
            # For unknown samples, labels should be -1
            assert torch.all(labels_known_idx[~is_known_target] == -1)
        
        if seen_known and seen_unknown:
            break 
    
    # Val set construction from train_base (80/20 split) ensures knowns.
    # Unknowns depend on their presence in train_base. CIFAR10 has all 10 classes in train.
    assert seen_known, "Validation set should contain known samples"
    assert seen_unknown, "Validation set should contain unknown samples from the original training classes"


def test_calibration_dataloader(base_config_dataset):
    """Test the calibration dataloader (which often is the val_dataloader)."""
    dm = OpenSetDataModule(base_config_dataset)
    dm.prepare_data()
    dm.setup(stage='validate') # Calibration typically uses validation set

    loader = dm.calibration_dataloader()
    assert loader is not None
    # Further checks would be similar to test_val_dataloader_batch
    _, _, is_known_target = next(iter(loader))
    assert is_known_target.shape[0] > 0 # Check if it yields batches