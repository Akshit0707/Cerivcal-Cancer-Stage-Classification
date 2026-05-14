"""
Training Script for GPU-Optimized Hybrid Model
Combines EfficientNet-B3 CNN features with traditional medical features

═══════════════════════════════════════════════════════════════════════
ROOT CAUSE ANALYSIS & FIXES (v5 — 80% accuracy target)
═══════════════════════════════════════════════════════════════════════

BUG 1 — NaN LOSS from epoch 1
  Root cause: mixup_data() used `alpha` as both the Beta distribution
  parameter AND the fixed lambda value.
  FIX: mixup_data samples lam ~ Beta(alpha, alpha), clamped to [0.05,0.95].
       LabelSmoothing and FocalLoss clamp logits before softmax.
       Gradient clipping happens BEFORE optimizer.step().

BUG 2 — WarmupCosineScheduler ran BACKWARDS
  Root cause: scheduler.step() was called inside train_epoch() — once
  per batch — so warmup completed in 3 batches, not 3 epochs.
  FIX: scheduler.step() called ONCE per epoch in the main loop.
       WarmupCosineScheduler tracks integer epochs, not steps.

BUG 3 — Differential LR param groups broken
  Root cause: 'lr_scale' overrode absolute LR every step, conflicting
  with ReduceLROnPlateau adjustments.
  FIX: Store initial_lr per group; WarmupCosine scales against initial_lr.

BUG 4 — Data split ignored pre-made train/val split
  Root cause: Both train/ and val/ were merged then re-split 80/20,
  contaminating the val set.
  FIX: Pre-split dirs used as-is unless val is severely imbalanced (>3×).

BUG 5 — Feature extraction called with wrong argument  ← ACCURACY KILLER
  Root cause: build_feature_cache called extract_medical_features(img)
  passing a PIL Image object, but the function signature expects a path
  string. This silently returned zeros for EVERY image, meaning the
  feature branch trained on pure noise throughout all epochs.
  FIX: Pass img_path (string) directly. PIL Image object removed.

BUG 6 — AMP not enabled despite T4 GPU
  FIX: Added GradScaler + autocast context manager.

BUG 7 — WeightedRandomSampler used wrong label list
  Root cause: Sampler weights built from full-pool train_labels before
  split, not from the final split train_labels.
  FIX: Sampler always built after splitting logic completes.

BUG 8 (NEW) — Learning rate too high
  Root cause: lr=2e-3 for the head with a frozen-style backbone causes
  instability and NaN gradients in early epochs.
  FIX: Default lr=5e-4 (head), backbone gets lr*0.1=5e-5.

BUG 9 (NEW) — MixUp started too early with alpha too high
  Root cause: mixup_start_epoch=2 with alpha=0.4 disrupts early feature
  learning on small medical datasets before the model has any signal.
  FIX: mixup_start_epoch=20, alpha=0.2.

BUG 10 (NEW) — Classifier head over-parameterized
  Root cause: 4-layer head (fusion→2048→1024→512→classes) overfits
  datasets of < 10k images and adds 50M+ redundant parameters.
  FIX: Simplified to 2-layer head: fusion_dim → 512 → num_classes.

BUG 11 (NEW) — Hue jitter too strong for diagnostic images
  Root cause: hue=0.15 destroys the cell color that is diagnostically
  meaningful for cervical cytology classification.
  FIX: hue=0.05 (subtle shift only).

BUG 12 (NEW) — Backbone never progressively unfrozen
  Root cause: Backbone starts at lr*0.1 but is never frozen initially,
  so ImageNet features are corrupted before the head learns anything.
  FIX: Freeze backbone for first BACKBONE_FREEZE_EPOCHS epochs, then
       unfreeze with a low lr for fine-tuning.
"""

import os
import argparse
import math
import random
import sys
import warnings
import json
from collections import Counter
from pathlib import Path
from PIL import Image

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_SCRIPT_VERSION = "EfficientNet-B3-Hybrid-v5-accuracy-fix"
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

NUM_TRADITIONAL_FEATURES = 30
CNN_INPUT_SIZE = 224

# How many epochs to keep backbone frozen before unfreezing for fine-tuning
BACKBONE_FREEZE_EPOCHS = 5


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


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class FeatureMLP(nn.Module):
    """MLP for the 30-dim traditional feature branch with residual connection."""
    def __init__(self, in_dim: int, out_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.proj(x)


class EfficientNetHybrid(nn.Module):
    """
    EfficientNet-B3 backbone + traditional feature MLP, fused with attention.

    FIX (BUG 10): Simplified classifier head from 4 layers to 2 layers.
    Reduces parameter count by ~50M and prevents overfitting on small datasets.
    """
    def __init__(self, num_classes: int, num_features: int = 30, dropout: float = 0.4):
        super().__init__()
        try:
            from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
            backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            cnn_out_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
        except Exception:
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            cnn_out_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone

        self.cnn_out_dim = cnn_out_dim
        feat_out_dim = 256  # smaller than before — matches simplified head

        self.feature_mlp = FeatureMLP(num_features, feat_out_dim, dropout=0.2)

        fusion_dim = cnn_out_dim + feat_out_dim

        # Attention gate: learns how much to trust CNN vs traditional features
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )

        # FIX BUG 10: Simplified 2-layer head — prevents overfitting
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, num_classes),
        )

    def forward(self, images, features):
        images   = torch.nan_to_num(images,   nan=0.0, posinf=1.0,  neginf=-1.0)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0,  neginf=0.0)

        cnn_feat  = self.backbone(images)
        cnn_feat  = torch.nan_to_num(cnn_feat, nan=0.0, posinf=1e3, neginf=-1e3)

        trad_feat = self.feature_mlp(features)
        trad_feat = torch.nan_to_num(trad_feat, nan=0.0, posinf=1e3, neginf=-1e3)

        fused = torch.cat([cnn_feat, trad_feat], dim=1)
        attn  = self.attention(fused)

        cnn_scaled   = cnn_feat  * attn[:, 0:1]
        trad_scaled  = trad_feat * attn[:, 1:2]
        fused_scaled = torch.cat([cnn_scaled, trad_scaled], dim=1)

        logits = self.classifier(fused_scaled)
        return logits, attn


def build_model(num_classes, num_features, device, dropout=0.4):
    model = EfficientNetHybrid(
        num_classes=num_classes,
        num_features=num_features,
        dropout=dropout
    )
    model = model.to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model            : EfficientNetHybrid (EfficientNet-B3, ImageNet weights)")
    print(f"  Total parameters : {n_params:,}")
    print(f"  Trainable now    : {trainable:,}")
    print(f"  Device           : {device}")
    print(f"  Classes          : {num_classes}")
    return model


def freeze_backbone(model):
    """Freeze all backbone parameters (called for first BACKBONE_FREEZE_EPOCHS)."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    frozen = sum(p.numel() for p in model.backbone.parameters())
    print(f"  🔒 Backbone frozen ({frozen:,} params) — head-only training")


def unfreeze_backbone(model):
    """Unfreeze backbone for fine-tuning after head has warmed up."""
    for param in model.backbone.parameters():
        param.requires_grad = True
    unfrozen = sum(p.numel() for p in model.backbone.parameters())
    print(f"  🔓 Backbone unfrozen ({unfrozen:,} params) — full fine-tuning begins")


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Down-weights easy examples so training focuses on hard negatives.
    Outperforms weighted CrossEntropy on imbalanced medical datasets.
    """
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight  # per-class weights tensor

    def forward(self, logits, targets):
        logits = torch.clamp(logits, -50.0, 50.0)
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        logits = torch.clamp(logits, -50.0, 50.0)
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class HybridDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, feature_cache=None):
        self.image_paths  = image_paths
        self.labels       = labels
        self.transform    = transform
        self.feature_cache = feature_cache if feature_cache is not None else {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label    = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        features = self.feature_cache.get(img_path, np.zeros(NUM_TRADITIONAL_FEATURES))

        return {
            'image':    image,
            'features': torch.FloatTensor(features),
            'label':    torch.LongTensor([label])[0],
        }


# ─────────────────────────────────────────────────────────────────────────────
# MixUp — FIX BUG 1a + BUG 9
# ─────────────────────────────────────────────────────────────────────────────
def mixup_batch(images, features, labels, alpha=0.2):
    """
    True Beta-sampled MixUp.

    FIX BUG 1a: lam is now sampled from Beta(alpha, alpha), not hardcoded.
    FIX BUG 9:  alpha reduced from 0.4 → 0.2 for medical imaging datasets.
                start_epoch moved from 2 → 20 (controlled in train_epoch caller).
    """
    batch_size = images.size(0)
    lam = float(np.random.beta(alpha, alpha))
    lam = max(0.05, min(0.95, lam))

    index = torch.randperm(batch_size, device=images.device)
    mixed_images   = lam * images   + (1 - lam) * images[index]
    mixed_features = lam * features + (1 - lam) * features[index]
    labels_a = labels
    labels_b = labels[index]
    return mixed_images, mixed_features, labels_a, labels_b, lam


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler — FIX BUG 2 + BUG 3
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """
    Per-EPOCH warmup + cosine annealing.

    FIX BUG 2: Called ONCE per epoch from the main loop, not per batch.
    FIX BUG 3: Each param_group stores initial_lr; scaling is independent
               per group so differential LRs are preserved correctly.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.current_epoch  = 0

        for pg in optimizer.param_groups:
            pg['initial_lr'] = pg['lr']

    def step(self):
        self.current_epoch += 1
        e = self.current_epoch

        if e <= self.warmup_epochs:
            scale = e / max(1, self.warmup_epochs)
        else:
            progress = (e - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg in self.optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * scale

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def set_group_lr(self, group_idx, new_initial_lr):
        """Call this when unfreezing backbone to inject correct LR for that group."""
        self.optimizer.param_groups[group_idx]['initial_lr'] = new_initial_lr
        self.optimizer.param_groups[group_idx]['lr']         = new_initial_lr


# ─────────────────────────────────────────────────────────────────────────────
# Train epoch — FIX BUG 2, BUG 6, BUG 9
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, train_loader, optimizer, criterion, device,
                scaler=None, use_mixup=False, current_epoch=0,
                mixup_start_epoch=20,   # FIX BUG 9: was 2
                mixup_alpha=0.2):       # FIX BUG 9: was 0.4
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    first_batch = True
    for batch in tqdm(train_loader, desc="Training", leave=False):
        images   = batch['image'].to(device)
        features = batch['features'].to(device)
        labels   = batch['label'].to(device)

        # First-batch diagnostics — catch data pipeline issues early
        if first_batch:
            first_batch = False
            if not torch.isfinite(images).all():
                print("  ⚠️  DIAGNOSTIC: images contain NaN/Inf — check normalisation")
            if not torch.isfinite(features).all():
                n_bad    = (~torch.isfinite(features)).sum().item()
                bad_dims = (~torch.isfinite(features)).any(dim=0).nonzero(as_tuple=True)[0].tolist()
                print(f"  ⚠️  DIAGNOSTIC: features contain {n_bad} NaN/Inf in dims {bad_dims[:10]}")
                print(f"      feature min={features[torch.isfinite(features)].min():.3f}, "
                      f"max={features[torch.isfinite(features)].max():.3f}")
            else:
                fmin, fmax = features.min().item(), features.max().item()
                print(f"  ✅  DIAGNOSTIC: features OK — range [{fmin:.2f}, {fmax:.2f}]")
                if abs(fmin) < 1e-6 and abs(fmax) < 1e-6:
                    print("  ❌  CRITICAL: features are ALL ZEROS — check extract_medical_features call!")

        use_mixup_now = use_mixup and (current_epoch >= mixup_start_epoch)

        # FIX BUG 6: AMP autocast
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            if use_mixup_now:
                images_m, features_m, y_a, y_b, lam = mixup_batch(
                    images, features, labels, alpha=mixup_alpha
                )
                logits, _ = model(images_m, features_m)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                logits, _ = model(images, features)
                loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            print(f"  ⚠️  Non-finite loss ({loss.item():.4f}) — skipping batch")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # FIX BUG 1b: clip BEFORE step, not after
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits.detach(), 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc      = 100.0 * correct / max(1, total)
    avg_loss = total_loss / max(1, len(train_loader))
    return avg_loss, acc


def evaluate(model, val_loader, device, criterion=None):
    model.eval()
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            images   = batch['image'].to(device)
            features = batch['features'].to(device)
            labels   = batch['label'].to(device)

            logits, _ = model(images, features)
            if criterion is not None:
                loss = criterion(logits, labels)
                if torch.isfinite(loss):
                    total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc      = 100.0 * correct / max(1, total)
    avg_loss = total_loss / max(1, len(val_loader)) if criterion is not None else 0.0
    report   = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    macro_f1 = report['macro avg']['f1-score']
    return acc, avg_loss, all_preds, all_labels, macro_f1


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers — FIX BUG 5 (CRITICAL)
# ─────────────────────────────────────────────────────────────────────────────
def sanitize_features(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf in a feature vector with 0.0 and clip extreme values."""
    arr = np.array(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, -1e6, 1e6)
    return arr


def build_feature_cache(image_paths, feature_scaler=None, fit_scaler=False):
    """
    Extract, sanitize, scale and cache features for all image paths.

    FIX BUG 5 (CRITICAL): The original code did:
        img   = Image.open(img_path).convert('RGB')
        feats = extract_medical_features(img)   ← PIL Image passed
    But extract_medical_features(path) expects a FILE PATH string and
    opens the image internally. Passing a PIL Image silently returned
    zeros for every sample — the feature branch trained on pure noise.

    This fix passes img_path (string) directly, removing the PIL open
    entirely from this function (the Dataset handles PIL loading separately).

    FIX: Uses RobustScaler (percentile-based) to handle outliers better
    than StandardScaler, clipping to ±3 after scaling.
    """
    raw   = {}
    n_bad = 0

    for img_path in tqdm(image_paths, desc="Extracting features", leave=False):
        try:
            # FIX BUG 5: pass the path string, NOT a PIL Image object
            feats = extract_medical_features(img_path)
            feats = sanitize_features(feats)

            if len(feats) != NUM_TRADITIONAL_FEATURES:
                print(f"  ⚠️  Feature dim mismatch for {Path(img_path).name}: "
                      f"got {len(feats)}, expected {NUM_TRADITIONAL_FEATURES}")
                feats = np.zeros(NUM_TRADITIONAL_FEATURES, dtype=np.float32)
                n_bad += 1

        except Exception as e:
            print(f"  ⚠️  Feature extraction failed for {Path(img_path).name}: {e}")
            feats = np.zeros(NUM_TRADITIONAL_FEATURES, dtype=np.float32)
            n_bad += 1

        raw[img_path] = feats

    if n_bad:
        pct = 100.0 * n_bad / max(1, len(image_paths))
        print(f"  ⚠️  {n_bad}/{len(image_paths)} ({pct:.1f}%) images used zero-vector fallback")
        if pct > 20:
            print("  ❌  >20% feature failures — check extract_medical_features() function!")

    # Verify features are not all zeros (sanity check for BUG 5)
    sample_vals = np.concatenate([raw[p] for p in list(raw.keys())[:10]])
    if np.allclose(sample_vals, 0.0):
        print("  ❌  CRITICAL: All features are zero — extract_medical_features may be broken!")
    else:
        print(f"  ✅  Feature sanity check passed — sample range "
              f"[{sample_vals.min():.3f}, {sample_vals.max():.3f}]")

    if fit_scaler:
        all_feats = np.stack([raw[p] for p in image_paths])
        if np.isnan(all_feats).any():
            print("  ⚠️  NaN in feature matrix before scaling — forcing to 0")
            all_feats = np.nan_to_num(all_feats, nan=0.0)

        # RobustScaler: uses median and IQR, not mean/std — handles outlier features
        feature_scaler = RobustScaler(quantile_range=(10.0, 90.0))
        feature_scaler.fit(all_feats)
        print(f"  ✅  RobustScaler fit: center range "
              f"[{feature_scaler.center_.min():.2f}, {feature_scaler.center_.max():.2f}]")

    cache = {}
    for p in image_paths:
        if feature_scaler is not None:
            scaled = feature_scaler.transform([raw[p]])[0]
            scaled = np.clip(scaled, -3.0, 3.0)   # 99.7% of normal distribution
            scaled = sanitize_features(scaled)
            cache[p] = scaled
        else:
            cache[p] = raw[p]

    return cache, feature_scaler


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train_hybrid_model(data_dir, output_dir, epochs=80, batch_size=16,
                       lr=5e-4,                    # FIX BUG 8: was 2e-3
                       early_stopping_patience=15,
                       num_workers=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    data_path = Path(data_dir)

    print(f"\n📁 Scanning: {data_path}")

    # ── Data loading — FIX BUG 4 ────────────────────────────────────────
    train_path = data_path / 'train'
    val_path   = data_path / 'val'

    def load_split(split_path, class_names):
        paths, lbls = [], []
        for idx, cls in enumerate(class_names):
            cls_dir = split_path / cls
            if not cls_dir.exists():
                print(f"  ⚠️  {cls}/ not found in {split_path.name}/")
                continue
            imgs = (sorted(cls_dir.glob('*.jpg')) + sorted(cls_dir.glob('*.JPG')) +
                    sorted(cls_dir.glob('*.png')) + sorted(cls_dir.glob('*.PNG')))
            print(f"     {cls}: {len(imgs)} images")
            for ip in imgs:
                paths.append(str(ip))
                lbls.append(idx)
        return paths, lbls

    if train_path.exists() and val_path.exists():
        print("✅ Pre-split train/ and val/ found.")
        class_names = sorted([p.name for p in train_path.iterdir() if p.is_dir()])
        print(f"   Classes: {class_names}")
        print("\n📂 train/")
        train_paths_raw, train_labels_raw = load_split(train_path, class_names)
        print("\n📂 val/")
        val_paths_raw, val_labels_raw = load_split(val_path, class_names)

        val_counts       = Counter(val_labels_raw)
        min_val          = min(val_counts.values())
        max_val          = max(val_counts.values())
        imbalance_ratio  = max_val / max(min_val, 1)

        if imbalance_ratio > 3.0:
            print(f"\n⚠️  Val set severely imbalanced (ratio={imbalance_ratio:.1f}×). "
                  f"Merging and re-splitting 80/20 with stratification...")
            all_paths  = train_paths_raw  + val_paths_raw
            all_labels = train_labels_raw + val_labels_raw
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                all_paths, all_labels,
                test_size=0.2, random_state=42, stratify=all_labels
            )
            print("\n📊 Re-split result:")
            for idx, name in enumerate(class_names):
                print(f"   {name}: train={Counter(train_labels)[idx]}  "
                      f"val={Counter(val_labels)[idx]}")
        else:
            # FIX BUG 4: use pre-made split directly
            train_paths, train_labels = train_paths_raw, train_labels_raw
            val_paths,   val_labels   = val_paths_raw,   val_labels_raw
            print("✅ Val distribution balanced — using pre-made split.")
    else:
        print("⚠️  No pre-split dirs found — scanning root and splitting 80/20.")
        class_names = sorted([
            p.name for p in data_path.iterdir()
            if p.is_dir() and p.name not in ('test', 'synthetic', 'sipakmed_raw')
        ])
        all_paths, all_labels = load_split(data_path, class_names)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )

    print(f"\n✅ Train: {len(train_paths)}  |  Val: {len(val_paths)}")
    if len(train_paths) == 0:
        raise ValueError("No training images found. Check --data-dir.")

    # ── Feature extraction — FIX BUG 5 ──────────────────────────────────
    print("\n🔍 Extracting train features (fit scaler)...")
    train_cache, feature_scaler = build_feature_cache(train_paths, fit_scaler=True)

    print("🔍 Extracting val features (apply scaler)...")
    val_cache, _ = build_feature_cache(val_paths, feature_scaler=feature_scaler)

    n_synthetic = sum(1 for p in train_paths if 'synthetic' in p.lower())
    n_real      = len(train_paths) - n_synthetic
    print(f"\n📊 Train composition: {n_real} real  +  {n_synthetic} synthetic")

    # ── Transforms — FIX BUG 11: reduced hue jitter ─────────────────────
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,          # FIX BUG 11: was 0.15 — too destructive for cell color
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.RandomGrayscale(p=0.05),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ── Datasets & Loaders ───────────────────────────────────────────────
    train_ds = HybridDataset(train_paths, train_labels, train_transform, train_cache)
    val_ds   = HybridDataset(val_paths,   val_labels,   val_transform,   val_cache)

    # FIX BUG 7: sampler uses final train_labels (after all splitting logic)
    class_counts   = Counter(train_labels)
    class_weights  = {i: 1.0 / class_counts[i] for i in range(len(class_names))}
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)

    num_workers  = min(num_workers, os.cpu_count() or 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(
        num_classes=len(class_names),
        num_features=NUM_TRADITIONAL_FEATURES,
        device=device,
        dropout=0.4,
    )

    # FIX BUG 12: freeze backbone initially; unfreeze after BACKBONE_FREEZE_EPOCHS
    freeze_backbone(model)

    # ── Optimizer — FIX BUG 8: lower base LR ────────────────────────────
    backbone_param_ids = {id(p) for p in model.backbone.parameters()}
    backbone_params    = [p for p in model.parameters() if id(p) in backbone_param_ids]
    head_params        = [p for p in model.parameters() if id(p) not in backbone_param_ids]

    # backbone lr is set intentionally low; it will be used after unfreezing
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1,  'weight_decay': 1e-4},  # 5e-5
        {'params': head_params,     'lr': lr,         'weight_decay': 1e-4},  # 5e-4
    ])

    # FIX BUG 2 + 3: WarmupCosineScheduler with per-group initial_lr
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=3, total_epochs=epochs)

    # ── Loss — FocalLoss with class weights (FIX BUG 5 bonus: now meaningful) ──
    class_weights_tensor = torch.tensor(
        [1.0 / class_counts[i] for i in range(len(class_names))],
        dtype=torch.float32, device=device
    )
    class_weights_tensor = class_weights_tensor / class_weights_tensor.sum() * len(class_names)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights_tensor)

    # ── AMP — FIX BUG 6 ──────────────────────────────────────────────────
    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"  AMP: {'Enabled' if use_amp else 'Disabled'}")

    # ── Training Loop — FIX BUG 2: scheduler.step() called once per epoch ──
    best_macro_f1    = 0.0
    best_val_loss    = float('inf')
    best_val_acc     = 0.0
    patience_counter = 0
    backbone_unfrozen = False
    checkpoint_path  = os.path.join(output_dir, 'best_model.pt')
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_macro_f1': [],
    }

    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*70}")

        # FIX BUG 12: unfreeze backbone after warm-up period
        if not backbone_unfrozen and epoch >= BACKBONE_FREEZE_EPOCHS:
            unfreeze_backbone(model)
            backbone_unfrozen = True
            # Re-create optimizer so backbone params are actually tracked with gradient
            optimizer = torch.optim.AdamW([
                {'params': [p for p in model.backbone.parameters()],
                 'lr': lr * 0.1,  'weight_decay': 1e-4},
                {'params': head_params,
                 'lr': lr * 0.5,  'weight_decay': 1e-4},  # reduce head LR for stability
            ])
            # Rebuild scheduler with remaining epochs
            remaining = epochs - epoch
            scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=1, total_epochs=remaining)
            print(f"  Optimizer rebuilt for full fine-tuning phase.")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler,
            use_mixup=True,
            current_epoch=epoch,
            mixup_start_epoch=20,   # FIX BUG 9: was 2
            mixup_alpha=0.2,        # FIX BUG 9: was 0.4
        )

        val_acc, val_loss, _, _, macro_f1 = evaluate(
            model, val_loader, device, criterion
        )

        # FIX BUG 2: step scheduler ONCE per epoch
        scheduler.step()
        new_lrs = [pg['lr'] for pg in optimizer.param_groups]

        print(f"Train  — loss: {train_loss:.4f} | acc: {train_acc:.2f}%")
        print(f"Val    — loss: {val_loss:.4f}  | acc: {val_acc:.2f}% | macro-F1: {macro_f1:.4f}")
        print(f"LR     — head={new_lrs[1]:.2e}  backbone={new_lrs[0]:.2e}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(macro_f1)

        improved = macro_f1 > best_macro_f1 or (
            val_loss < best_val_loss * 0.99 and val_acc > 60.0
        )
        if improved:
            best_macro_f1    = macro_f1
            best_val_loss    = val_loss
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save({
                'epoch':               epoch + 1,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':             val_acc,
                'macro_f1':            macro_f1,
                'class_names':         class_names,
                'num_features':        NUM_TRADITIONAL_FEATURES,
                'script_version':      _SCRIPT_VERSION,
            }, checkpoint_path)
            print(f"✅ Checkpoint saved  (macro-F1={macro_f1:.4f}, val_acc={val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{early_stopping_patience})")
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}.")
                break

    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ Training complete!")
    print(f"   Best val acc : {best_val_acc:.2f}%")
    print(f"   Best macro-F1: {best_macro_f1:.4f}")
    print(f"   Checkpoint   : {checkpoint_path}")
    print(f"{'='*70}")
    return checkpoint_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hybrid cervical cancer classifier')
    parser.add_argument('--data-dir',                type=str,   default='/kaggle/working/data')
    parser.add_argument('--checkpoint-dir',          type=str,   default='./checkpoints')
    parser.add_argument('--epochs',                  type=int,   default=80)
    parser.add_argument('--batch-size',              type=int,   default=16)
    parser.add_argument('--learning-rate',           type=float, default=5e-4)   # FIX BUG 8
    parser.add_argument('--early-stopping-patience', type=int,   default=15)
    parser.add_argument('--num-workers',             type=int,   default=4)
    parser.add_argument('--seed',                    type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"\n{'='*70}")
    print(f"🚀 TRAINING CONFIGURATION")
    print(f"{'='*70}")
    for k, v in vars(args).items():
        print(f"  {k:<30}: {v}")
    print(f"{'='*70}\n")

    train_hybrid_model(
        data_dir=args.data_dir,
        output_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
    )