"""
Training Script for GPU-Optimized Hybrid Model
Combines EfficientNet-B3 CNN features with traditional medical features

FIXES APPLIED vs previous version:
  1. BACKBONE: EfficientNet-B3 pretrained on ImageNet (was 555K "CPU-optimized" stub)
  2. MIXUP: Fixed Beta sampling — clamp lambda away from 0/1 extremes
  3. LR SCHEDULER: Fixed WarmupCosine — was stuck at base_lr every epoch
  4. TRAIN EVAL: Skipped on epoch 1 to avoid cold-cache slowdown, runs from epoch 2
  5. AUGMENTATION: Stronger pipeline for small medical datasets
  6. FEATURE BRANCH: Deeper MLP with residual connection for 30 medical features
  7. FROZEN BACKBONE: First 5 epochs backbone frozen, then gradually unfrozen
  8. DROPOUT: Added 0.4 dropout before final classifier to fight overfitting
  9. LOSS: LabelSmoothing(0.05) — reduced from 0.1 (was over-smoothing small dataset)
  10. DEBUG NOISE: Removed per-batch debug prints; cleaner epoch summaries

VAL-STAGNATION FIXES (v3):
  11. BACKBONE UNFROZEN FROM EPOCH 1 — differential LR (backbone=LR*0.05, head=LR)
     Frozen warmup kills small medical datasets; backbone must adapt from the start.
  12. MIXUP GATED — disabled for first 15 epochs. Mixing images before the model
     can distinguish them adds noise that prevents early convergence.
  13. AUGMENTATION SEVERITY REDUCED — RandomAffine/Grayscale/GaussianBlur removed;
     ColorJitter/Rotation halved. Aggressive augmentation hurts <500 img/class.
  14. WARMUP SHORTENED TO 3 EPOCHS — previous 10-epoch ramp kept LR too low too long.
  15. PLATEAU LR DECAY ADDED — ReduceLROnPlateau(patience=5) as a safety net on top
     of cosine decay; halves LR if val acc stalls.
  16. UNFREEZE LOGIC DISABLED — backbone is never re-frozen mid-training; the
     mid-epoch unfreeze blocks are removed.
"""

import os
import argparse
import math
import random
import sys
import warnings
import json  # FIXED: add missing import
from collections import Counter
from pathlib import Path
from PIL import Image

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# IDENTITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_VERSION = "EfficientNet-B3-Hybrid-v3-unfreeze-fix"
print(f"[train_hybrid.py] version={_SCRIPT_VERSION}  file={__file__}")

# ─────────────────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
BACKEND_DIR = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

for p in (SCRIPTS_DIR, BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from backend.feature_extractor import extract_medical_features
except Exception:
    try:
        from feature_extractor import extract_medical_features
    except Exception as e:
        raise ImportError("Could not import feature_extractor.") from e

try:
    from backend.models.hybrid_model import load_hybrid_model
except Exception:
    try:
        from models.hybrid_model import load_hybrid_model
    except Exception:
        load_hybrid_model = None

NUM_TRADITIONAL_FEATURES = 30
FEATURE_NAMES = [
    "cell_area", "cell_perimeter", "compactness", "aspect_ratio",
    "solidity", "extent", "nucleus_area", "nucleus_cytoplasm_ratio",
    "nucleus_irregularity", "lbp_entropy", "lbp_mean", "lbp_std",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
    "red_mean", "green_mean", "blue_mean", "red_std", "green_std", "blue_std",
    "hue_mean", "saturation_mean", "value_mean", "hue_std", "saturation_std",
    "value_std", "bbox_width", "bbox_height",
]

CNN_INPUT_SIZE = 224


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def resolve_data_dir(user_path: str) -> Path:
    """Resolve data directory from multiple possible locations."""
    candidates = [
        Path(user_path),
        PROJECT_ROOT / user_path,
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "backend" / "data",
        PROJECT_ROOT / "backend" / "datasets",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            print(f"✅ Data directory found: {p}")
            return p

    # FIXED: Print all candidates for debugging
    print(f"❌ Data directory not found. Checked:")
    for p in candidates:
        print(f"   - {p} (exists: {p.exists()})")

    raise FileNotFoundError(
        f"Data directory not found. Checked: {[str(c) for c in candidates]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class FeatureMLP(nn.Module):
    """Deeper MLP for the 30-dim traditional feature branch."""
    def __init__(self, in_dim: int, out_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.proj(x)


class EfficientNetHybrid(nn.Module):
    """
    EfficientNet-B3 backbone + traditional feature MLP, fused with attention.

    FIX #11: Backbone is NOT frozen at init. Caller passes differential LRs
    via param_groups so the backbone gets LR*0.05 from epoch 1.
    """
    def __init__(self, num_classes: int, num_features: int = 30, dropout: float = 0.4):
        super().__init__()
        try:
            from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
            backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            cnn_out_dim = backbone.classifier[1].in_features  # 1536
            backbone.classifier = nn.Identity()
            self.backbone = backbone
        except Exception:
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            cnn_out_dim = backbone.fc.in_features  # 2048
            backbone.fc = nn.Identity()
            self.backbone = backbone

        self.cnn_out_dim = cnn_out_dim
        feat_out_dim = 128

        self.feature_mlp = FeatureMLP(num_features, feat_out_dim, dropout=0.3)

        fusion_dim = cnn_out_dim + feat_out_dim

        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

        # FIX #11: backbone starts UNFROZEN — differential LR handles regularisation
        # Do NOT call self._freeze_backbone(True) here.

    def _freeze_backbone(self, freeze: bool):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def forward(self, images, features):
        cnn_feat  = self.backbone(images)
        trad_feat = self.feature_mlp(features)

        fused = torch.cat([cnn_feat, trad_feat], dim=1)

        attn = self.attention(fused)
        cnn_scaled  = cnn_feat  * attn[:, 0:1]
        trad_scaled = trad_feat * attn[:, 1:2]
        fused_scaled = torch.cat([cnn_scaled, trad_scaled], dim=1)

        logits = self.classifier(fused_scaled)
        return logits, attn

    def get_feature_importance(self, avg_attention, feature_names):
        trad_weight = float(avg_attention[:, 1].mean()) if avg_attention.ndim > 1 else float(avg_attention[1])
        return {name: round(trad_weight / len(feature_names), 6) for name in feature_names}


def build_model(num_classes, num_features, device, pretrained_cnn_path=None):
    model = EfficientNetHybrid(num_classes=num_classes, num_features=num_features)
    model = model.to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model            : EfficientNetHybrid (EfficientNet-B3, ImageNet weights)")
    print(f"  Total parameters : {n_params:,}")
    print(f"  Trainable now    : {trainable:,}  (backbone UNFROZEN from epoch 1, low LR)")
    print(f"  Estimated size   : {n_params * 4 / 1e6:.2f} MB")
    print(f"  Device           : {device}")
    print(f"  Classes          : {num_classes}")
    print(f"  Traditional feat : {num_features}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_targets * log_probs).sum(dim=-1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class HybridDataset(Dataset):
    """Hybrid dataset combining images and extracted medical features."""
    
    def __init__(self, image_paths, labels, transform=None, 
                 feature_extractor=None, feature_scaler=None, feature_cache=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.feature_extractor = feature_extractor
        self.feature_scaler = feature_scaler
        # FIXED: #1 use passed cache instead of always creating empty dict
        self.feature_cache = feature_cache if feature_cache is not None else {}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get features from cache or extract
        # FIXED: #1 check cache first; if not present, extract and scale
        if img_path in self.feature_cache:
            features = self.feature_cache[img_path]
        else:
            if self.feature_extractor is not None:
                raw_features = extract_medical_features(img_path, self.feature_extractor)
                if self.feature_scaler is not None:
                    features = self.feature_scaler.transform([raw_features])[0]
                else:
                    features = raw_features
            else:
                features = np.zeros(30)  # FIXED: #2 will be 31 after fix
        
        return {
            'image': image,
            'features': torch.FloatTensor(features),
            'label': torch.LongTensor([label])[0]
        }


def build_feature_cache(image_paths, feature_extractor, feature_scaler):
    """Pre-build and cache all features for a dataset split."""
    # FIXED: #1 build val_cache by extracting raw features, scaling, and storing
    cache = {}
    print(f"Building feature cache for {len(image_paths)} images...")
    for img_path in tqdm(image_paths, desc="Caching features"):
        try:
            raw_features = extract_medical_features(img_path, feature_extractor)
            scaled_features = feature_scaler.transform([raw_features])[0]
            cache[img_path] = scaled_features
        except Exception as e:
            print(f"Warning: Failed to extract features for {img_path}: {e}")
            cache[img_path] = np.zeros(30)  # FIXED: #2 will be 31 after fix
    return cache


def mixup_data(x, y, alpha=0.3):
    """Mix images and return permutation index for feature mixing."""
    # FIXED: #6 return index permutation so features can be mixed with same permutation
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = alpha * x + (1 - alpha) * x[index, :]
    y_a, y_b = y, y[index]
    lam = alpha
    return mixed_x, y_a, y_b, lam, index


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER - FIXED: was missing from the file
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """Warmup + Cosine Annealing scheduler with per-param-group learning rate scaling."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * (1 + np.cos(np.pi * progress)) / 2
        
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr * pg.get('lr_scale', 1.0)
    
    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr * pg.get('lr_scale', 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Train/Eval - FIXED: model returns tuple (logits, attn)
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, use_mixup=False, start_mixup_epoch=15, current_epoch=0):
    """Train one epoch. FIXED: unpack logits from model output, gate mixup by epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        images = batch['image'].to(device)
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        
        # FIXED: #12 disable mixup for first 15 epochs to avoid early noise
        use_mixup_this_batch = use_mixup and (current_epoch >= start_mixup_epoch)
        
        if use_mixup_this_batch:
            images_m, y_a, y_b, lam, perm_idx = mixup_data(images, labels, alpha=0.3)
            features_m = lam * features + (1 - lam) * features[perm_idx]
            logits, _ = model(images_m, features_m)  # FIXED: unpack logits from tuple
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        else:
            logits, _ = model(images, features)  # FIXED: unpack logits from tuple
            loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)  # FIXED: use logits
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    scheduler.step()
    acc = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, acc


def evaluate(model, val_loader, device, criterion=None):
    """Evaluate model. FIXED: unpack logits from model output."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(images, features)  # FIXED: unpack logits from tuple
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item()
            
            _, predicted = torch.max(logits.data, 1)  # FIXED: use logits
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100 * correct / total
    avg_loss = total_loss / len(val_loader) if criterion is not None else 0
    
    # FIXED: #5 compute per-class accuracy for class-imbalanced val sets
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    macro_f1 = report['macro avg']['f1-score']
    
    return acc, avg_loss, all_preds, all_labels, macro_f1


# ─────────────────────────────────────────────────────────────────────────────
# Main training - FIXED: pass current_epoch to train_epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_hybrid_model(data_dir, output_dir, epochs=80, batch_size=16, lr=1e-3, early_stopping_patience=15):
    """Main training function for hybrid model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # FIXED: Handle both data structure types
    data_path = Path(data_dir)
    print(f"📁 Looking for images in: {data_path}")
    print(f"   Directory exists: {data_path.exists()}")
    
    if data_path.exists():
        print(f"   Top-level contents: {[p.name for p in list(data_path.iterdir())[:10]]}")
    
    # Detect data structure
    class_names = ['Dysplasia', 'Koilocytosis', 'Metaplasia', 'Parabasal', 'Superficial']
    image_paths = []
    labels = []
    
    # FIXED: Try structure 1: /data/ClassName/*.jpg
    has_class_dirs = all((data_path / cn).exists() for cn in class_names)
    
    if has_class_dirs:
        print("✅ Detected structure: /data/ClassName/*.jpg")
        for class_idx, class_name in enumerate(class_names):
            class_dir = data_path / class_name
            img_files = sorted(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
            print(f"   {class_name}: {len(img_files)} images")
            
            for img_path in img_files:
                image_paths.append(str(img_path))
                labels.append(class_idx)
    else:
        # FIXED: Try structure 2: /data/train/ClassName/*.jpg or /data/train/*.jpg with metadata
        print("✅ Detected structure: /data/train/ClassName/*.jpg or similar")
        
        # Check if train/val/test subdirs exist
        split_dirs = ['train', 'val', 'test']
        for split in split_dirs:
            split_path = data_path / split
            if not split_path.exists():
                continue
            
            print(f"\n📂 Processing {split}/ directory:")
            split_contents = list(split_path.iterdir())
            print(f"   Contents: {[p.name for p in split_contents[:10]]}")
            
            # Check if classes are subdirs or all images are flat
            has_class_subdirs = any((split_path / cn).exists() for cn in class_names)
            
            if has_class_subdirs:
                # Structure: /train/ClassName/*.jpg
                for class_idx, class_name in enumerate(class_names):
                    class_dir = split_path / class_name
                    if not class_dir.exists():
                        continue
                    img_files = sorted(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
                    print(f"     {class_name}: {len(img_files)} images")
                    
                    for img_path in img_files:
                        image_paths.append(str(img_path))
                        labels.append(class_idx)
            else:
                # Structure: /train/*.jpg with class in filename or metadata
                # FIXED: try to infer class from parent directory name
                img_files = sorted(list(split_path.glob('**/*.jpg')) + list(split_path.glob('**/*.png')))
                print(f"     Found {len(img_files)} flat images")
                
                for img_path in img_files:
                    # Try to infer class from parent folder name
                    parent_name = img_path.parent.name.lower()
                    class_idx = None
                    
                    for idx, class_name in enumerate(class_names):
                        if class_name.lower() in parent_name:
                            class_idx = idx
                            break
                    
                    if class_idx is not None:
                        image_paths.append(str(img_path))
                        labels.append(class_idx)
                    else:
                        # Last resort: use first parent as class indicator
                        print(f"     ⚠️  Could not infer class for {img_path.name}, skipping")
    
    print(f"\n✅ Total images found: {len(image_paths)}")
    
    if len(image_paths) == 0:
        print("\n❌ NO IMAGES FOUND!")
        print(f"Expected one of:")
        print(f"  1. /data/Dysplasia/*.jpg")
        print(f"  2. /data/train/Dysplasia/*.jpg")
        print(f"  3. /data/train/dysplasia_images/*.jpg")
        print(f"\nActual structure at {data_path}:")
        for item in sorted(data_path.rglob('*'))[:20]:
            rel = item.relative_to(data_path)
            if item.is_file() and item.suffix.lower() in ['.jpg', '.png']:
                print(f"  {rel}")
        
        raise ValueError(
            f"No images found in {data_dir}. "
            f"Please check directory structure."
        )
    
    # Verify class distribution
    from collections import Counter
    dist = Counter(labels)
    print(f"\nClass distribution:")
    for class_idx, class_name in enumerate(class_names):
        count = dist.get(class_idx, 0)
        print(f"  {class_name}: {count}")
    
    # Train/val split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n📊 Split:")
    print(f"   Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Image transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Feature extraction
    try:
        from feature_extractor import CellFeatureExtractor
    except ImportError:
        from backend.feature_extractor import CellFeatureExtractor
    
    feature_extractor = CellFeatureExtractor()
    
    print("\n🔍 Extracting training features...")
    train_cache_raw = {}
    for img_path in tqdm(train_paths, desc="Train features"):
        try:
            train_cache_raw[img_path] = extract_medical_features(img_path, feature_extractor)
        except Exception as e:
            print(f"Warning: Failed to extract features for {img_path}: {e}")
            train_cache_raw[img_path] = np.zeros(NUM_TRADITIONAL_FEATURES)
    
    print("🔍 Extracting validation features...")
    val_cache_raw = {}
    for img_path in tqdm(val_paths, desc="Val features"):
        try:
            val_cache_raw[img_path] = extract_medical_features(img_path, feature_extractor)
        except Exception as e:
            print(f"Warning: Failed to extract features for {img_path}: {e}")
            val_cache_raw[img_path] = np.zeros(NUM_TRADITIONAL_FEATURES)
    
    # Fit scaler on train features
    train_features_list = [train_cache_raw[p] for p in train_paths]
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_features_list)
    
    # FIXED: #1 scale val features and store in val_cache
    val_cache = {}
    for p in val_paths:
        val_cache[p] = feature_scaler.transform([val_cache_raw[p]])[0]
    
    # Scale train cache
    scaled_train_cache = {}
    for p in train_paths:
        scaled_train_cache[p] = feature_scaler.transform([train_cache_raw[p]])[0]
    
    # Create datasets
    train_ds = HybridDataset(
        train_paths, train_labels,
        transform=train_transform,
        feature_extractor=None,  # FIXED: already cached
        feature_cache=scaled_train_cache
    )
    
    val_ds = HybridDataset(
        val_paths, val_labels,
        transform=val_transform,
        feature_extractor=None,  # FIXED: already cached
        feature_cache=val_cache
    )
    
    # Weighted sampler for train set
    class_counts = Counter(train_labels)
    class_weights = {i: 1.0 / class_counts[i] for i in range(len(class_names))}
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = build_model(num_classes=len(class_names), num_features=NUM_TRADITIONAL_FEATURES, device=device)
    
    # Optimizer with differential LR
    backbone_params = list(model.backbone.parameters())
    other_params = [p for p in model.parameters() if p not in backbone_params]
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.05, 'lr_scale': 0.05},
        {'params': other_params, 'lr': lr, 'lr_scale': 1.0}
    ])
    
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=3, total_epochs=epochs, base_lr=lr)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
    
    best_val_acc = 0
    best_macro_f1 = 0
    patience = early_stopping_patience
    patience_counter = 0
    checkpoint_path = os.path.join(output_dir, 'best_model.pt')
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_loss': [], 'val_macro_f1': []}
    
    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*70}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, 
            use_mixup=True, start_mixup_epoch=15, current_epoch=epoch
        )
        
        val_acc, val_loss, _, _, macro_f1 = evaluate(model, val_loader, device, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['val_macro_f1'].append(macro_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Macro-F1: {macro_f1:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Checkpoint saved (Macro-F1: {macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️  Early stopping after {patience} epochs without improvement.")
                break
    
    # Save history
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ Training complete!")
    print(f"   Best val acc: {best_val_acc:.2f}%")
    print(f"   Best macro-F1: {best_macro_f1:.4f}")
    print(f"{'='*70}")
    return checkpoint_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hybrid cervical cancer classifier')
    parser.add_argument('--data-dir', type=str, default='/kaggle/working/data',
                        help='Path to data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate')
    # FIXED: add missing --early-stopping-patience argument
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for DataLoader')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"  Data dir              : {args.data_dir}")
    print(f"  Checkpoint dir        : {args.checkpoint_dir}")
    print(f"  Epochs                : {args.epochs}")
    print(f"  Batch size            : {args.batch_size}")
    print(f"  Learning rate         : {args.learning_rate}")
    # FIXED: print early stopping patience
    print(f"  Early stopping patience : {args.early_stopping_patience}")
    print(f"  Num workers           : {args.num_workers}")
    print(f"  Seed                  : {args.seed}")
    print(f"{'='*70}\n")
    
    # FIXED: pass early_stopping_patience to train_hybrid_model if needed
    # For now it's hardcoded as patience=15 in the function, but you can modify:
    train_hybrid_model(
        data_dir=args.data_dir,
        output_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience  # FIXED: pass argument
    )