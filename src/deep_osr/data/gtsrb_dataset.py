import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from PIL import Image
import numpy as np

class GTSRBDataset(Dataset):
    """
    German Traffic Sign Recognition Benchmark Dataset
    Contains 40 classes of German traffic signs + unknown traffic signs from other countries
    """
    
    # German traffic signs (GTSRB) - these are our known classes
    GTSRB_CLASSES = list(range(43))  # GTSRB has 43 classes (0-42)
    
    # We'll select 40 of them as known and keep 3 as unknown for demonstration
    KNOWN_CLASSES = list(range(40))  # Classes 0-39 as known
    UNKNOWN_CLASSES = list(range(40, 43))  # Classes 40-42 as unknown (simulating other countries)
    
    def __init__(self, root, split='train', transform=None, download=True):
        self.root = root
        self.split = split
        self.transform = transform
        
        if download:
            self._download()
            
        self._load_data()
        
        # Create targets list for compatibility with OpenSetDataset
        self.targets = [label for _, label in self.samples]
    
    def _download(self):
        """Download GTSRB dataset"""
        os.makedirs(self.root, exist_ok=True)
        
        # GTSRB training data
        train_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
        
        # GTSRB test data  
        test_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
        test_labels_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"
        
        train_dir = os.path.join(self.root, "GTSRB", "Final_Training")
        test_dir = os.path.join(self.root, "GTSRB", "Final_Test")
        
        if not os.path.exists(train_dir):
            print("Downloading GTSRB training data...")
            download_and_extract_archive(train_url, self.root, remove_finished=True)
            
        if not os.path.exists(test_dir):
            print("Downloading GTSRB test data...")
            download_and_extract_archive(test_url, self.root, remove_finished=True)
            download_and_extract_archive(test_labels_url, self.root, remove_finished=True)
    
    def _load_data(self):
        """Load and organize the dataset"""
        self.samples = []
        
        if self.split == 'train':
            self._load_train_data()
        else:
            self._load_test_data()
    
    def _load_train_data(self):
        """Load training data from GTSRB structure"""
        train_dir = os.path.join(self.root, "GTSRB", "Final_Training", "Images")
        
        for class_id in range(43):  # GTSRB has 43 classes
            class_dir = os.path.join(train_dir, f"{class_id:05d}")
            
            if os.path.exists(class_dir):
                # Read CSV file with image annotations
                csv_file = os.path.join(class_dir, f"GT-{class_id:05d}.csv")
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file, sep=';')
                        
                        for _, row in df.iterrows():
                            img_path = os.path.join(class_dir, row['Filename'])
                            if os.path.exists(img_path):
                                self.samples.append((img_path, class_id))
                    except Exception as e:
                        print(f"Error reading CSV {csv_file}: {e}")
                        # Fallback: load all images in the directory
                        for img_file in os.listdir(class_dir):
                            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
                                img_path = os.path.join(class_dir, img_file)
                                self.samples.append((img_path, class_id))
    
    def _load_test_data(self):
        """Load test data from GTSRB structure"""
        test_dir = os.path.join(self.root, "GTSRB", "Final_Test", "Images")
        
        # Read test labels (check both possible locations)
        labels_file = os.path.join(self.root, "GT-final_test.csv")
        if not os.path.exists(labels_file):
            labels_file = os.path.join(self.root, "GTSRB", "GT-final_test.csv")
        test_loaded = False
        
        if os.path.exists(labels_file):
            try:
                df = pd.read_csv(labels_file, sep=';')
                
                for _, row in df.iterrows():
                    img_path = os.path.join(test_dir, row['Filename'])
                    class_id = row['ClassId']
                    
                    if os.path.exists(img_path):
                        self.samples.append((img_path, class_id))
                        test_loaded = True
            except Exception as e:
                print(f"Error reading test labels {labels_file}: {e}")
        
        if not test_loaded:
            # Fallback: use a subset of training data as test
            print("Using fallback: splitting training data for test set")
            train_dataset = GTSRBDataset(self.root, split='train', download=False)
            # Use last 20% of training samples as test
            test_size = len(train_dataset.samples) // 5
            self.samples = train_dataset.samples[-test_size:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class GTSRBUnknownDataset(Dataset):
    """
    Dataset for unknown traffic signs (simulated as non-German traffic signs)
    For demonstration, we'll use some online traffic sign images or create synthetic ones
    """
    
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.unknown_root = os.path.join(root, "unknown_signs")
        
        # Create some synthetic unknown signs by taking a few GTSRB signs 
        # and marking them as unknown (simulating other countries)
        self._create_unknown_samples()
        
    def _create_unknown_samples(self):
        """Create or load unknown traffic sign samples"""
        os.makedirs(self.unknown_root, exist_ok=True)
        
        # For now, we'll simulate unknown signs by using some GTSRB images
        # but treating them as unknown. In a real scenario, you'd have
        # actual traffic signs from other countries.
        
        # We can use classes 40-42 from GTSRB as "unknown" 
        # or use some other images if available
        gtsrb_dir = os.path.join(self.root, "GTSRB", "Final_Training", "Images")
        
        self.samples = []
        
        # Use classes 40-42 as unknown classes
        unknown_class_ids = [40, 41, 42]
        
        for class_id in unknown_class_ids:
            class_dir = os.path.join(gtsrb_dir, f"{class_id:05d}")
            
            if os.path.exists(class_dir):
                csv_file = os.path.join(class_dir, f"GT-{class_id:05d}.csv")
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file, sep=';')
                    
                    # Take only a subset of images from these classes
                    for i, (_, row) in enumerate(df.iterrows()):
                        if i >= 100:  # Limit to 100 images per unknown class
                            break
                            
                        img_path = os.path.join(class_dir, row['Filename'])
                        if os.path.exists(img_path):
                            # Label as -1 for unknown
                            self.samples.append((img_path, -1))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label 