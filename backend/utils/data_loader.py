import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

class CervicalCancerDataset(Dataset):
    """Custom Dataset for cervical cancer images"""
    
    def __init__(self, data_dir, transform=None, split='train'):
        """
        Args:
            data_dir: Root directory containing class subdirectories
            transform: Optional transform to be applied on images
            split: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.classes = ['Normal', 'CIN1', 'CIN2', 'CIN3', 'Cancer']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load all images
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
                for img_path in class_dir.glob('*.jpeg'):
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"Loaded {len(self.images)} images for {split} set")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(augment=True):
    """Get image transformations"""
    if augment:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """Create train, validation, and test data loaders"""
    
    train_dataset = CervicalCancerDataset(
        data_dir, 
        transform=get_transforms(augment=True), 
        split='train'
    )
    
    val_dataset = CervicalCancerDataset(
        data_dir, 
        transform=get_transforms(augment=False), 
        split='val'
    )
    
    test_dataset = CervicalCancerDataset(
        data_dir, 
        transform=get_transforms(augment=False), 
        split='test'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on macOS
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on macOS
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on macOS
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader
