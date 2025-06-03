import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl
from typing import List, Optional, Tuple
import numpy as np
from .transforms import get_transforms

class OpenSetDataset(Dataset):
    def __init__(self, base_dataset, known_classes: List[int], unknown_classes: List[int],
                 split: str = "train", transform=None, include_unknown_in_train=False):
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform
        self.split = split # 'train', 'val', 'test'
        
        self.known_classes = sorted(list(set(known_classes)))
        self.unknown_classes = sorted(list(set(unknown_classes)))

        self.known_class_map = {original_label: new_label for new_label, original_label in enumerate(self.known_classes)}
        
        self.samples = [] # List of (data_idx_in_base, new_label, is_known)

        for i in range(len(base_dataset)):
            try:
                # torchvision datasets return (image, label)
                _, original_label = base_dataset[i]
            except IndexError: # Some datasets might not be indexable like this before loading all data
                print(f"Warning: Could not get label for index {i} directly. Ensure base_dataset is compatible.")
                continue

            if original_label in self.known_classes:
                new_label = self.known_class_map[original_label]
                self.samples.append((i, new_label, True)) # True for is_known
            elif original_label in self.unknown_classes:
                if split == "train" and not include_unknown_in_train:
                    continue # Skip unknowns for training set if not desired
                # For val/test, unknowns are included. Label can be -1 or a specific unknown marker.
                self.samples.append((i, -1, False)) # False for is_known, label -1 for unknown
            # Else: class is neither known nor unknown, so it's ignored (held-out completely)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_idx, new_label, is_known = self.samples[idx]
        image, _ = self.base_dataset[base_idx] # Get image using base_dataset index
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(new_label, dtype=torch.long), torch.tensor(is_known, dtype=torch.bool)


class OpenSetDataModule(pl.LightningDataModule):
    def __init__(self, cfg_dataset):
        super().__init__()
        self.cfg = cfg_dataset
        self.train_transform, self.test_transform = get_transforms(self.cfg.name, self.cfg.img_size)

        self.train_dataset_full: Optional[Dataset] = None
        self.val_dataset_full: Optional[Dataset] = None # Contains both knowns and unknowns for val
        self.test_dataset_known: Optional[Dataset] = None
        self.test_dataset_unknown: Optional[Dataset] = None
        
        # For calibration set
        self.train_dataset_for_model: Optional[Dataset] = None
        self.val_dataset_for_calibration: Optional[Dataset] = None


    def prepare_data(self):
        # Download data if needed
        if self.cfg.name == "cifar10":
            CIFAR10(self.cfg.path, train=True, download=True)
            CIFAR10(self.cfg.path, train=False, download=True)
        # Add other datasets here if needed

    def _filter_dataset_by_original_classes(self, dataset, target_classes):
        indices = [i for i, (_, label) in enumerate(dataset) if label in target_classes]
        return Subset(dataset, indices)

    def setup(self, stage: Optional[str] = None):
        if self.cfg.name == "cifar10":
            train_base = CIFAR10(self.cfg.path, train=True) # transform applied later
            test_base = CIFAR10(self.cfg.path, train=False)  # transform applied later
        else:
            raise ValueError(f"Dataset {self.cfg.name} not supported yet.")

        # Split base train_data into train and val
        # Example: 80% train, 20% val from the original training set
        # This val set will contain samples from known_classes and unknown_classes as per their distribution in train_base
        train_len = int(len(train_base) * 0.8)
        val_len = len(train_base) - train_len
        train_subset_base, val_subset_base = random_split(train_base, [train_len, val_len])
        
        # Training set: only known classes
        # Filter train_subset_base to only include known_classes
        train_known_indices = [i for i in train_subset_base.indices if train_base.targets[i] in self.cfg.known_classes]
        train_subset_for_model_base = Subset(train_base, train_known_indices)
        self.train_dataset_full = OpenSetDataset(train_subset_for_model_base, self.cfg.known_classes, [], "train", self.train_transform)

        # Validation set: mix of known and unknown (as available in val_subset_base)
        self.val_dataset_full = OpenSetDataset(val_subset_base, self.cfg.known_classes, self.cfg.unknown_classes, "val", self.test_transform)

        # Test sets: separate known and unknown
        self.test_dataset_known = OpenSetDataset(
            self._filter_dataset_by_original_classes(test_base, self.cfg.known_classes),
            self.cfg.known_classes, [], "test", self.test_transform
        )
        self.test_dataset_unknown = OpenSetDataset(
            self._filter_dataset_by_original_classes(test_base, self.cfg.unknown_classes),
            [], self.cfg.unknown_classes, "test", self.test_transform # No known classes here
        )
        
        # For calibration: split val_dataset_full further if needed, or use a portion of train_dataset_full
        # Or, more simply, use a fraction of the val_dataset_full
        # For now, let's assume calibration happens on the val_dataset_full
        # Or, create a dedicated calibration set from the original training data not used for model training or val
        # This part can be refined based on specific calibration strategy. For simplicity, we'll use val_dataset_full for calibration.

        print(f"Train dataset size: {len(self.train_dataset_full)}")
        print(f"Validation dataset size: {len(self.val_dataset_full)}")
        print(f"Test Known dataset size: {len(self.test_dataset_known)}")
        print(f"Test Unknown dataset size: {len(self.test_dataset_unknown)}")


    def train_dataloader(self):
        return DataLoader(self.train_dataset_full, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True, persistent_workers=True if self.cfg.num_workers > 0 else False)

    def val_dataloader(self):
        # This dataloader will provide batches with mixed known/unknown samples for OSR evaluation during training
        return DataLoader(self.val_dataset_full, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)

    def test_dataloader(self):
        # For eval.py, we want separate evaluation on knowns and unknowns, then combine results.
        # So eval.py will call model.predict() on test_dataset_known and test_dataset_unknown separately.
        # This dataloader can return a list of dataloaders or be adapted.
        # For Lightning's trainer.test(), it expects one or a list.
        # Let's make it so eval.py handles this. Here, provide combined for generic .test()
        # However, the blueprint implies eval.py runs on specific txt files.
        # So, this test_dataloader might be used for a final combined test metric report.
        # For now, let's make it return the val_dataloader's structure for consistency.
        # A more advanced setup would pass specific dataloaders to eval.py.
        
        # Combining test_known and test_unknown for a single test pass if needed by pl.Trainer.test()
        # Note: labels for unknown in combined_test_dataset will be -1.
        if self.test_dataset_known and self.test_dataset_unknown and len(self.test_dataset_unknown)>0:
            combined_test_dataset = torch.utils.data.ConcatDataset([self.test_dataset_known, self.test_dataset_unknown])
            return DataLoader(combined_test_dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)
        elif self.test_dataset_known:
            return DataLoader(self.test_dataset_known, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)
        else: # Should not happen in OSR if we have unknowns
            return None


    # Dataloader for calibration set (can be a subset of validation or training)
    def calibration_dataloader(self):
        # Use the validation set for calibration for simplicity here
        # A more rigorous approach might use a dedicated calibration set
        return DataLoader(self.val_dataset_full, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)