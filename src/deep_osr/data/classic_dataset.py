import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl
from typing import Optional, List
import numpy as np
from PIL import Image # Added import

from .transforms import get_transforms
from .gtsrb_dataset import GTSRBDataset # Assuming this is a standard dataset class

class ClassicDataModule(pl.LightningDataModule):
    def __init__(self, cfg_dataset):
        super().__init__()
        self.cfg = cfg_dataset
        self.train_transform, self.test_transform = get_transforms(self.cfg.name, self.cfg.img_size)

        self.num_training_classes = self.cfg.num_classes # e.g. 8 known classes
        self.total_original_classes = self.cfg.total_original_classes # e.g. 10 for CIFAR10
        self.known_classes_original_ids = list(self.cfg.known_classes_original_ids)
        self.unknown_classes_original_ids = list(self.cfg.unknown_classes_original_ids)

        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset_known: Optional[torch.utils.data.Dataset] = None
        self.val_dataset_unknown: Optional[torch.utils.data.Dataset] = None
        self.test_dataset_known: Optional[torch.utils.data.Dataset] = None
        self.test_dataset_unknown: Optional[torch.utils.data.Dataset] = None

        # Mapping for known class labels (0 to N-1)
        self.known_class_map = {original_id: new_id for new_id, original_id in enumerate(self.known_classes_original_ids)}

    def prepare_data(self):
        # Download data if needed
        if self.cfg.name == "cifar10":
            CIFAR10(self.cfg.path, train=True, download=True)
            CIFAR10(self.cfg.path, train=False, download=True)
        elif self.cfg.name == "gtsrb":
            # Assuming GTSRBDataset handles its own download if necessary
            GTSRBDataset(self.cfg.path, split='train', download=True)
            GTSRBDataset(self.cfg.path, split='test', download=True)
        # Add other datasets here

    def setup(self, stage: Optional[str] = None):
        if self.cfg.name == "cifar10":
            # Load base datasets without transforms; TransformedDataset will handle them.
            train_base_full = CIFAR10(self.cfg.path, train=True, download=True) 
            test_base_full = CIFAR10(self.cfg.path, train=False, download=True)
        elif self.cfg.name == "gtsrb":
            # Load base datasets without transforms; TransformedDataset will handle them.
            train_base_full = GTSRBDataset(self.cfg.path, split='train', download=True) # Removed transform
            test_base_full = GTSRBDataset(self.cfg.path, split='test', download=True)   # Removed transform
        else:
            raise ValueError(f"Dataset {self.cfg.name} not supported yet.")

        # Split train_base_full into train and validation sets based on original class IDs
        if self.cfg.get("val_split_ratio", 0.2) > 0:
            train_len = int(len(train_base_full) * (1.0 - self.cfg.val_split_ratio))
            val_len = len(train_base_full) - train_len
            # Ensure stratification if possible, or at least preserve class distributions for knowns
            # For simplicity, using random_split. For more complex scenarios, consider group-wise splitting.
            train_subset_indices, val_subset_indices = random_split(range(len(train_base_full)), [train_len, val_len])
            
            train_base_for_model_setup = Subset(train_base_full, list(train_subset_indices.indices))
            val_base_for_eval_setup = Subset(train_base_full, list(val_subset_indices.indices))
        else:
            train_base_for_model_setup = train_base_full
            # Fallback validation: use a subset of the test set if no val_split_ratio
            val_base_for_eval_setup = Subset(test_base_full, list(range(min(len(test_base_full), self.cfg.get("val_fallback_test_samples", 1000)))))

        # Create training dataset (only known classes, with mapped labels)
        self.train_dataset = self._create_filtered_dataset(
            train_base_for_model_setup, 
            self.known_classes_original_ids, 
            transform=self.train_transform, 
            map_labels=True
        )

        # Create validation datasets (known and unknown, original labels for eval)
        self.val_dataset_known = self._create_filtered_dataset(
            val_base_for_eval_setup, 
            self.known_classes_original_ids, 
            transform=self.test_transform, 
            map_labels=False # Keep original labels for eval
        )
        self.val_dataset_unknown = self._create_filtered_dataset(
            val_base_for_eval_setup, 
            self.unknown_classes_original_ids, 
            transform=self.test_transform, 
            map_labels=False # Keep original labels for eval
        )

        # Create test datasets (known and unknown, original labels for eval)
        self.test_dataset_known = self._create_filtered_dataset(
            test_base_full, 
            self.known_classes_original_ids, 
            transform=self.test_transform, 
            map_labels=False
        )
        self.test_dataset_unknown = self._create_filtered_dataset(
            test_base_full, 
            self.unknown_classes_original_ids, 
            transform=self.test_transform, 
            map_labels=False
        )

        print(f"Number of training classes (known): {self.num_training_classes}")
        print(f"Total original classes: {self.total_original_classes}")
        print(f"Known original class IDs: {self.known_classes_original_ids}")
        print(f"Unknown original class IDs: {self.unknown_classes_original_ids}")
        print(f"Train dataset size (knowns only, mapped labels): {len(self.train_dataset)}")
        print(f"Validation dataset size (knowns): {len(self.val_dataset_known)}")
        print(f"Validation dataset size (unknowns): {len(self.val_dataset_unknown)}")
        print(f"Test dataset size (knowns): {len(self.test_dataset_known)}")
        print(f"Test dataset size (unknowns): {len(self.test_dataset_unknown)}")

    def _create_filtered_dataset(self, base_dataset_subset, target_original_classes: List[int], transform, map_labels: bool):
        indices = []
        original_labels_for_subset = [] # Store original labels for these indices

        # If base_dataset_subset is a Subset, access its underlying dataset and indices
        if isinstance(base_dataset_subset, Subset):
            underlying_dataset = base_dataset_subset.dataset
            subset_indices = base_dataset_subset.indices
        else:
            underlying_dataset = base_dataset_subset
            subset_indices = list(range(len(underlying_dataset)))

        for i in subset_indices:
            # Accessing labels depends on the dataset type (e.g., CIFAR10 uses .targets, GTSRB might use .labels or a method)
            if hasattr(underlying_dataset, 'targets'): # For CIFAR10
                original_label = underlying_dataset.targets[i]
            elif hasattr(underlying_dataset, 'labels'): # For GTSRB or similar
                original_label = underlying_dataset.labels[i]
            elif hasattr(underlying_dataset, 'samples'): # For ImageFolder-like datasets
                 _, original_label = underlying_dataset.samples[i]
            else: # Fallback: try getting the item and inferring label (less efficient)
                try:
                    _, original_label = underlying_dataset[i]
                except Exception as e:
                    raise ValueError(f"Could not get label for dataset {type(underlying_dataset)}. Error: {e}")
            
            if original_label in target_original_classes:
                indices.append(i) # Store the index from the underlying_dataset
                original_labels_for_subset.append(original_label)
        
        return TransformedDataset(underlying_dataset, indices, original_labels_for_subset, transform, self.known_class_map if map_labels else None)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True, persistent_workers=True if self.cfg.num_workers > 0 else False)

    def val_dataloader(self):
        # For standard validation during training, use only knowns with mapped labels if model expects that.
        # However, for the requested evaluation, we need separate known/unknown dataloaders.
        # The PL `val_dataloader` is typically for metrics logged during training.
        # We will create separate dataloaders for the custom evaluation step.
        # For now, let's assume val_dataloader provides knowns for standard PL validation loop.
        # This might need adjustment based on how ClassicLightningModule handles validation.
        # If ClassicLightningModule's validation_step expects original labels or mixed data, this needs to change.
        # For simplicity, let's provide a val_dataloader with knowns, similar to train, but with test_transform.
        
        # Create a temporary validation set from val_base_for_eval_setup for knowns with mapped labels for PL trainer
        # This is if the trainer's validation loop needs mapped labels for knowns.
        temp_val_known_mapped = self._create_filtered_dataset(
            self.val_dataset_known.base_dataset, # Corrected: Access base_dataset directly
            self.known_classes_original_ids,
            transform=self.test_transform,
            map_labels=True
        )
        if len(temp_val_known_mapped) == 0:
             print("Warning: Validation dataloader (knowns, mapped) is empty. PL validation metrics might be affected.")
             # Provide a dummy dataloader to prevent errors if it's truly empty, though this indicates a setup issue.
             return DataLoader(torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0)), batch_size=self.cfg.batch_size)
        return DataLoader(temp_val_known_mapped, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, persistent_workers=True if self.cfg.num_workers > 0 else False)

    def test_dataloader(self):
        # Similar to val_dataloader, PL test_dataloader is for standard testing.
        # We'll use custom dataloaders for the specific evaluation.
        # For now, provide a test_dataloader with knowns, similar to val.
        temp_test_known_mapped = self._create_filtered_dataset(
            self.test_dataset_known.base_dataset, # Corrected: Access base_dataset directly
            self.known_classes_original_ids,
            transform=self.test_transform,
            map_labels=True
        )
        if len(temp_test_known_mapped) == 0:
            print("Warning: Test dataloader (knowns, mapped) is empty.")
            return DataLoader(torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0)), batch_size=self.cfg.batch_size)
        return DataLoader(temp_test_known_mapped, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, persistent_workers=True if self.cfg.num_workers > 0 else False)

    # Custom dataloaders for the specific evaluation requested
    def eval_val_known_dataloader(self):
        if len(self.val_dataset_known) == 0:
            print("Warning: Eval Val Known Dataloader is empty.")
            return DataLoader(torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0)), batch_size=self.cfg.batch_size)
        return DataLoader(self.val_dataset_known, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)

    def eval_val_unknown_dataloader(self):
        if len(self.val_dataset_unknown) == 0:
            print("Warning: Eval Val Unknown Dataloader is empty.")
            return DataLoader(torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0)), batch_size=self.cfg.batch_size)
        return DataLoader(self.val_dataset_unknown, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)

    def eval_test_known_dataloader(self):
        if len(self.test_dataset_known) == 0:
            print("Warning: Eval Test Known Dataloader is empty.")
            return DataLoader(torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0)), batch_size=self.cfg.batch_size)
        return DataLoader(self.test_dataset_known, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)

    def eval_test_unknown_dataloader(self):
        if len(self.test_dataset_unknown) == 0:
            print("Warning: Eval Test Unknown Dataloader is empty.")
            return DataLoader(torch.utils.data.TensorDataset(torch.empty(0), torch.empty(0)), batch_size=self.cfg.batch_size)
        return DataLoader(self.test_dataset_unknown, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)

# Helper Dataset class to apply transforms and label mapping
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices: List[int], original_labels: List[int], transform, known_class_map: Optional[dict] = None):
        self.base_dataset = base_dataset
        self.indices = indices # Indices relative to base_dataset
        self.original_labels = original_labels # Original labels corresponding to self.indices
        self.transform = transform
        self.known_class_map = known_class_map # e.g., {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}

        if len(self.indices) != len(self.original_labels):
            raise ValueError("Indices and original_labels must have the same length.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        original_label = self.original_labels[idx]
        
        # Get image from base_dataset. How to get only image depends on base_dataset structure
        if hasattr(self.base_dataset, 'data') and hasattr(self.base_dataset, 'targets'): # CIFAR-like
            image_data = self.base_dataset.data[base_idx]
            if isinstance(image_data, np.ndarray):
                # Assuming HWC format for CIFAR numpy array
                image = Image.fromarray(image_data) 
            else: # If it's not numpy, it might be already a PIL image or other format
                image = image_data
            
            # Now, if 'image' is a PIL Image, check its mode
            if hasattr(image, 'mode') and image.mode != 'RGB':
                image = image.convert('RGB')

        elif hasattr(self.base_dataset, 'loader') and hasattr(self.base_dataset, 'samples'): # ImageFolder-like
            path, _ = self.base_dataset.samples[base_idx]
            image = self.base_dataset.loader(path)
        else: # Generic fallback, assuming base_dataset[idx] returns (image, label)
            try:
                image, _ = self.base_dataset[base_idx]
            except Exception as e:
                raise RuntimeError(f"Could not retrieve image from base_dataset at index {base_idx}. Error: {e}")

        if self.transform:
            image = self.transform(image)

        if self.known_class_map:
            # Map to new label if it's a known class, otherwise this sample shouldn't be here
            # or this dataset is for evaluation with original labels
            if original_label not in self.known_class_map:
                 # This case should ideally be filtered out before creating this dataset if it's for training
                 # Or, if it's for evaluation, known_class_map should be None.
                raise ValueError(f"Original label {original_label} not in known_class_map during label mapping.")
            label = self.known_class_map[original_label]
        else:
            label = original_label # Use original label (e.g., for evaluation datasets)
            
        return image, label

