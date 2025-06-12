import torch
import torch.utils.data as data
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10
from typing import Optional, List
import os
from omegaconf import DictConfig

from deep_osr.data.gtsrb_dataset import GTSRBDataset
from deep_osr.data.transforms import get_transforms


class EnhancedOpenSetDataset(data.Dataset):
    """
    Enhanced dataset for open-set recognition that includes unknown samples in training.
    Supports training with dummy class labels for unknown samples.
    """
    
    def __init__(self, base_dataset, known_classes: List[int], unknown_classes: List[int],
                 split: str = "train", transform=None, include_unknown_in_train=True,
                 unknown_sample_ratio=0.1):
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform
        self.split = split
        self.include_unknown_in_train = include_unknown_in_train
        self.unknown_sample_ratio = unknown_sample_ratio

        self.known_classes = sorted(list(set(known_classes)))
        self.unknown_classes = sorted(list(set(unknown_classes)))

        self.known_class_map = {original_label: new_label for new_label, original_label in enumerate(self.known_classes)}
        
        self.samples = []  # List of (data_idx_in_base, new_label, is_known)

        # Collect all samples
        known_samples = []
        unknown_samples = []
        
        for i in range(len(base_dataset)):
            try:
                _, original_label = base_dataset[i]
            except IndexError:
                continue
                
            if original_label in self.known_classes:
                new_label = self.known_class_map[original_label]
                known_samples.append((i, new_label, True))
            elif original_label in self.unknown_classes:
                # For unknown samples, we'll use a special label (num_known_classes)
                unknown_samples.append((i, len(self.known_classes), False))

        # Add all known samples
        self.samples.extend(known_samples)
        
        # Add unknown samples based on split and inclusion policy
        if split == "train" and include_unknown_in_train:
            # Include a subset of unknown samples for training with dummy class
            num_unknown_to_include = int(len(known_samples) * unknown_sample_ratio)
            if num_unknown_to_include > 0 and len(unknown_samples) > 0:
                # Randomly sample unknown examples
                unknown_indices = torch.randperm(len(unknown_samples))[:num_unknown_to_include]
                selected_unknowns = [unknown_samples[i] for i in unknown_indices]
                self.samples.extend(selected_unknowns)
        elif split in ["val", "test"]:
            # Include all unknown samples for validation and testing
            self.samples.extend(unknown_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_idx, label, is_known = self.samples[idx]
        image, _ = self.base_dataset[data_idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, is_known


class EnhancedOpenSetDataModule(pl.LightningDataModule):
    """
    Enhanced data module for open-set recognition with dummy class support.
    """
    
    def __init__(self, cfg_dataset: DictConfig):
        super().__init__()
        self.cfg = cfg_dataset
        self.train_transform, self.test_transform = get_transforms(self.cfg.name, self.cfg.img_size)

        self.train_dataset_full: Optional[data.Dataset] = None
        self.val_dataset_full: Optional[data.Dataset] = None
        self.test_dataset_known: Optional[data.Dataset] = None
        self.test_dataset_unknown: Optional[data.Dataset] = None

    def prepare_data(self):
        # Download data if needed
        if self.cfg.name == "cifar10":
            CIFAR10(self.cfg.path, train=True, download=True)
            CIFAR10(self.cfg.path, train=False, download=True)
        elif self.cfg.name == "gtsrb":
            GTSRBDataset(self.cfg.path, split='train', download=True)
            GTSRBDataset(self.cfg.path, split='test', download=True)

    def _filter_dataset_by_original_classes(self, dataset, target_classes):
        indices = [i for i, (_, label) in enumerate(dataset) if label in target_classes]
        return Subset(dataset, indices)

    def setup(self, stage: Optional[str] = None):
        if self.cfg.name == "cifar10":
            train_base = CIFAR10(self.cfg.path, train=True)
            test_base = CIFAR10(self.cfg.path, train=False)
        elif self.cfg.name == "gtsrb":
            train_base = GTSRBDataset(self.cfg.path, split='train')
            test_base = GTSRBDataset(self.cfg.path, split='test')
        else:
            raise ValueError(f"Dataset {self.cfg.name} not supported yet.")

        # Split base train data into train and val
        train_len = int(len(train_base) * 0.8)
        val_len = len(train_base) - train_len
        train_subset_base, val_subset_base = random_split(train_base, [train_len, val_len])

        # Training set: include both known and unknown samples for dummy class training
        include_unknowns = self.cfg.get('include_unknown_in_train', True)
        unknown_ratio = self.cfg.get('unknown_sample_ratio', 0.1)
        
        self.train_dataset_full = EnhancedOpenSetDataset(
            train_subset_base, 
            self.cfg.known_classes_original_ids, 
            self.cfg.unknown_classes_original_ids, 
            "train", 
            self.train_transform,
            include_unknown_in_train=include_unknowns,
            unknown_sample_ratio=unknown_ratio
        )

        # Validation set: mix of known and unknown
        self.val_dataset_full = EnhancedOpenSetDataset(
            val_subset_base, 
            self.cfg.known_classes_original_ids, 
            self.cfg.unknown_classes_original_ids, 
            "val", 
            self.test_transform,
            include_unknown_in_train=True  # Always include unknowns in validation
        )

        # Test sets: separate known and unknown for detailed evaluation
        self.test_dataset_known = EnhancedOpenSetDataset(
            self._filter_dataset_by_original_classes(test_base, self.cfg.known_classes_original_ids),
            self.cfg.known_classes_original_ids, [], "test", self.test_transform
        )
        self.test_dataset_unknown = EnhancedOpenSetDataset(
            self._filter_dataset_by_original_classes(test_base, self.cfg.unknown_classes_original_ids),
            [], self.cfg.unknown_classes_original_ids, "test", self.test_transform
        )

        print(f"Train dataset size: {len(self.train_dataset_full)}")
        print(f"Validation dataset size: {len(self.val_dataset_full)}")
        print(f"Test Known dataset size: {len(self.test_dataset_known)}")
        print(f"Test Unknown dataset size: {len(self.test_dataset_unknown)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset_full, 
            batch_size=self.cfg.batch_size, 
            shuffle=True, 
            num_workers=self.cfg.num_workers, 
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset_full, 
            batch_size=self.cfg.batch_size, 
            num_workers=self.cfg.num_workers, 
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False
        )

    def test_dataloader(self):
        # Combined test set for standard lightning testing
        if self.test_dataset_known and self.test_dataset_unknown and len(self.test_dataset_unknown) > 0:
            combined_test_dataset = torch.utils.data.ConcatDataset([self.test_dataset_known, self.test_dataset_unknown])
            return DataLoader(combined_test_dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)
        elif self.test_dataset_known:
            return DataLoader(self.test_dataset_known, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)
        else:
            return None

    def calibration_dataloader(self):
        # Use validation set for calibration
        return DataLoader(
            self.val_dataset_full, 
            batch_size=self.cfg.batch_size, 
            num_workers=self.cfg.num_workers, 
            pin_memory=True
        )
