"""
Training Script for GPU-Optimized Hybrid Model
Combines CNN features with traditional medical features

Features:
- Automatic feature extraction from images
- GPU-optimized training with mixed precision (AMP)
- Early stopping and learning rate scheduling
- Feature importance analysis
- Comprehensive evaluation metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import argparse
import sys
import time
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class BalancedLoss(nn.Module):
    """Balanced Cross Entropy Loss with moderate class weights"""
    def __init__(self, alpha=None):
        super(BalancedLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.alpha)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing for better generalization"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for hard example mining - focuses training on hard-to-classify samples
    Great for minority classes like Cancer where model struggles
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss  # Down-weight easy examples
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Add backend directory to path (for models/ and feature_extractor.py)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.hybrid_model import create_hybrid_model, load_hybrid_model
from feature_extractor import extract_medical_features


class HybridDataset(Dataset):
    """
    Dataset that loads images and extracts traditional features on-the-fly
    """
    def __init__(self, data_dir, transform=None, feature_cache=None, feature_scaler=None):
        """
        Args:
            data_dir: Path to data directory with class subdirectories
            transform: Image transformations
            feature_cache: Pre-computed features (optional, for speed)
            feature_scaler: Sklearn StandardScaler for feature normalization
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.feature_cache = feature_cache if feature_cache is not None else {}
        self.feature_scaler = feature_scaler
        
        # Find all images
        self.samples = []
        self.class_to_idx = {}
        
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx
            
            for img_path in class_dir.glob('*.png'):
                # Skip macOS hidden files
                if not img_path.name.startswith('._'):
                    self.samples.append((str(img_path), idx))
            for img_path in class_dir.glob('*.jpg'):
                # Skip macOS hidden files
                if not img_path.name.startswith('._'):
                    self.samples.append((str(img_path), idx))
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        print(f"Found {len(self.samples)} images in {len(self.class_to_idx)} classes")
        
        # Print class distribution
        class_counts = {}
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Extract traditional features (cached if available)
        if img_path in self.feature_cache:
            features = self.feature_cache[img_path]
        else:
            features = extract_medical_features(image)
            self.feature_cache[img_path] = features
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Normalize features using scaler
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform([features])[0]
        
        # Convert features to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        return image, features_tensor, label


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam, criterion):
    """Mixup loss with custom criterion (supports Focal Loss or CE)"""
    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    return lam * loss_a + (1 - lam) * loss_b


def get_transforms(augment=True):
    """Get image transformations with better augmentation"""
    if augment:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomGrayscale(p=0.1),
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


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam, class_weights):
    """Mixup loss"""
    loss_a = F.cross_entropy(pred, y_a, weight=class_weights, label_smoothing=0.1)
    loss_b = F.cross_entropy(pred, y_b, weight=class_weights, label_smoothing=0.1)
    return lam * loss_a + (1 - lam) * loss_b


def train_epoch(model, train_loader, optimizer, device, epoch, criterion, scaler=None):
    """Train for one epoch with optional mixed precision"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = (device.type == 'cuda' and scaler is not None)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, features, labels in pbar:
        images = images.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass with Mixup augmentation
        optimizer.zero_grad()
        
        # Apply Mixup to images
        images_mixed, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)
        
        # Mixed precision forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                logits, attention_weights = model(images_mixed, features)
                loss = mixup_criterion(logits, labels_a, labels_b, lam, criterion)
        else:
            logits, attention_weights = model(images_mixed, features)
            loss = mixup_criterion(logits, labels_a, labels_b, lam, criterion)
        
        # Debug first batch
        if total == 0:
            print(f"\nDEBUG - Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
            print(f"DEBUG - Mixup lambda: {lam:.2f}")
            if use_amp:
                print("DEBUG - Using AMP (Automatic Mixed Precision)")
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"\\nWARNING: NaN loss! Skipping batch.")
            continue
        
        # Backward pass with gradient clipping
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_attention_weights = []
    
    with torch.no_grad():
        for images, features, labels in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass with traditional features
            logits, attention_weights = model(images, features)
            loss = F.cross_entropy(logits, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attention_weights.append(attention_weights.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    # Average attention weights across all validation samples
    avg_attention = np.concatenate(all_attention_weights, axis=0).mean(axis=0)
    
    return epoch_loss, epoch_acc, all_preds, all_labels, avg_attention


def train_hybrid_model(args):
    """Main training function"""
    
    print("="*80)
    print("CPU-Optimized Hybrid Model Training")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu')
    print(f"\nUsing device: {device}")
    
    if device.type == 'cpu':
        print("Training on CPU - optimized for efficiency")
        # Enable CPU optimizations
        torch.set_num_threads(args.num_threads)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Define transforms
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    
    # First pass: collect all features for normalization
    print("Extracting features for normalization...")
    temp_dataset = HybridDataset(
        Path(args.data_dir) / 'train',
        transform=None
    )
    
    all_features = []
    for i in tqdm(range(len(temp_dataset)), desc="Collecting features"):
        _, features, _ = temp_dataset[i]
        all_features.append(features.numpy())
    
    # Fit scaler on training features
    feature_scaler = StandardScaler()
    feature_scaler.fit(all_features)
    print(f"Feature normalization fitted (mean: {feature_scaler.mean_[:3]}, std: {feature_scaler.scale_[:3]})")
    
    # Create datasets with scaler
    train_dataset = HybridDataset(
        Path(args.data_dir) / 'train',
        transform=train_transform,
        feature_scaler=feature_scaler
    )
    
    val_dataset = HybridDataset(
        Path(args.data_dir) / 'val',
        transform=val_transform,
        feature_scaler=feature_scaler
    )
    
    # Create balanced sampler with AGGRESSIVE oversampling
    class_counts = Counter([label for _, label in train_dataset.samples])
    
    print("\nClass distribution:")
    for cls_idx, cls_name in train_dataset.idx_to_class.items():
        count = class_counts[cls_idx]
        print(f"  {cls_name}: {count}")
    
    # AGGRESSIVE oversampling for minority classes
    max_count = max(class_counts.values())
    
    # Sample weights with stronger power (0.8 for aggressive minority sampling)
    sample_weights = [(max_count / class_counts[label]) ** 0.8 for _, label in train_dataset.samples]
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=int(len(train_dataset) * 2.0),  # 2x oversampling for minorities
        replacement=True
    )
    
    # GPU optimization: use more workers and persistent workers for faster data loading
    num_workers = args.num_workers if device.type == 'cuda' else 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # Use sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Create model
    print("\nCreating model...")
    num_classes = len(train_dataset.class_to_idx)
    num_features = 30  # Number of traditional features
    
    model = create_hybrid_model(
        num_classes=num_classes,
        num_traditional_features=num_features,
        pretrained_cnn_path=args.pretrained_cnn,
        device=device
    )
    
    # Use AGGRESSIVE class weights for minority class detection (CRITICAL for clinical use)
    class_weights = []
    for cls_name in sorted(class_counts.keys()):
        weight = (max_count / class_counts[cls_name]) ** 0.7  # AGGRESSIVE weighting for minorities
        class_weights.append(weight)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    
    # FOCAL LOSS for hard example mining (focuses on minority classes like Cancer)
    print("Using Focal Loss (gamma=2.5) - focuses on hard-to-classify examples")
    criterion = FocalLoss(alpha=class_weights, gamma=2.5)
    
    # GPU optimization: higher learning rate and create AMP scaler
    lr = args.learning_rate * 2.0 if device.type == 'cuda' else args.learning_rate
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
    
    # Create gradient scaler for AMP
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # ReduceLROnPlateau for adaptive learning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    if device.type == 'cuda':
        print(f"✓ GPU Training with AMP enabled (LR: {lr:.6f})")
    print("="*80)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, epoch, criterion, scaler
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, avg_attention = validate(
            model, val_loader, device
        )
        
        # Update learning rate based on validation accuracy
        scheduler.step(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'num_classes': num_classes,
                'num_traditional_features': num_features,
                'class_to_idx': train_dataset.class_to_idx,
                'avg_attention_weights': avg_attention
            }
            
            checkpoint_path = checkpoint_dir / 'best_hybrid_model.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.early_stopping_patience})")
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
        
        # Periodic checkpoints
        if epoch % 10 == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
            }, checkpoint_path)
            print(f"  Saved checkpoint at epoch {epoch}")
    
    # Final evaluation
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    
    # Load best model for final evaluation
    best_checkpoint_path = checkpoint_dir / 'best_hybrid_model.pth'
    model = load_hybrid_model(best_checkpoint_path, device=device)
    
    # Final validation
    print("\nFinal Evaluation on Validation Set:")
    val_loss, val_acc, val_preds, val_labels, avg_attention = validate(
        model, val_loader, device
    )
    
    # Classification report
    class_names = [train_dataset.idx_to_class[i] for i in range(num_classes)]
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=class_names))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(val_labels, val_preds)
    print(cm)
    
    # Feature importance analysis
    print("\n" + "="*80)
    print("Feature Importance Analysis")
    print("="*80)
    
    feature_names = [
        'cell_area', 'cell_perimeter', 'compactness', 'aspect_ratio', 
        'solidity', 'extent', 'nucleus_area', 'nc_ratio',
        'nucleus_irregularity', 'lbp_entropy', 'lbp_mean', 'lbp_std',
        'glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation',
        'r_mean', 'g_mean', 'b_mean', 'r_std', 'g_std', 'b_std',
        'h_mean', 's_mean', 'v_mean', 'h_std', 's_std', 'v_std',
        'convexity', 'circularity'
    ]
    
    importance_dict = model.get_feature_importance(avg_attention, feature_names)
    
    print("\nTop 10 Most Important Features:")
    for i, (name, score) in enumerate(list(importance_dict.items())[:10], 1):
        print(f"  {i:2d}. {name:25s}: {score:.4f}")
    
    print("\n✓ Training completed successfully!")
    print(f"Best model saved at: {best_checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train CPU-Optimized Hybrid Model')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='../data',
                       help='Path to data directory')
    
    # Model parameters
    parser.add_argument('--pretrained-cnn', type=str, default=None,
                       help='Path to pretrained CNN model (optional)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training (default 64 for GPU, use 16 for CPU)')
    parser.add_argument('--learning-rate', type=float, default=0.0003,
                       help='Base learning rate (will be doubled for GPU)')
    parser.add_argument('--early-stopping-patience', type=int, default=25,
                       help='Early stopping patience (increased for convergence)')
    
    # System parameters
    parser.add_argument('--cpu-only', action='store_true',
                       help='Force CPU training even if GPU is available')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (use 0 for CPU, 4+ for GPU)')
    parser.add_argument('--num-threads', type=int, default=4,
                       help='Number of CPU threads for training')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Train model
    train_hybrid_model(args)


if __name__ == '__main__':
    main()
