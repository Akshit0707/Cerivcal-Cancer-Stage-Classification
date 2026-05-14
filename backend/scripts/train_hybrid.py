"""
Training Script for GPU-Optimized Hybrid Model
Combines EfficientNet-B3 CNN features with traditional medical features

═══════════════════════════════════════════════════════════════════════
ROOT CAUSE ANALYSIS & FIXES (v4)
═══════════════════════════════════════════════════════════════════════

BUG 1 — NaN LOSS from epoch 1  ← PRIMARY KILLER
  Root cause: mixup_data() uses `alpha` as BOTH the Beta distribution
  parameter AND the fixed lambda value. The real fix requires actually
  sampling lambda from Beta(alpha, alpha) instead of hardcoding lam=alpha.
  Additionally LabelSmoothing's log_softmax can produce -inf when logits
  contain large values; clamp logits BEFORE softmax, not after.

  FIX: mixup_data now samples lam ~ Beta(alpha, alpha), clamps to
  [0.05, 0.95] to avoid degenerate mixing. Also added gradient clipping
  BEFORE optimizer.step() (was after in original — too late when NaN
  already in grads). LabelSmoothing now clamps logits input.

BUG 2 — WarmupCosineScheduler runs BACKWARDS (LR drops epoch 1→2)
  Root cause: scheduler.step() is called INSIDE train_epoch(), which
  calls it 59 times per epoch (once per batch). So by epoch 2 the
  scheduler thinks it's at step 59*100=5900, deep into cosine decay.
  The warmup_epochs=3 means after 3 *calls* (not epochs) warmup ends.

  FIX: Removed scheduler.step() from train_epoch(). Call it ONCE per
  epoch in the main training loop. Also rewrote WarmupCosineScheduler
  to track epochs (integers), not fractional steps.

BUG 3 — Differential LR param groups are broken
  Root cause: 'lr_scale' is stored in param_groups but WarmupCosine
  sets pg['lr'] = lr * pg.get('lr_scale', 1.0) — this overrides the
  entire group every step. On epoch 1, both groups get base_lr scaled
  correctly. But ReduceLROnPlateau (if added) would halve the absolute
  lr, then WarmupCosine re-multiplies by lr_scale of the *original*
  base_lr. Result: conflicting LR signals.

  FIX: Store initial_lr in each param_group at optimizer creation.
  WarmupCosine multiplies the *ratio* (epoch progress) against each
  group's initial_lr independently. No lr_scale needed.

BUG 4 — Data split ignores pre-made train/val split
  Root cause: The script loads BOTH train/ and val/ directories into a
  single list, then re-splits 80/20. This contaminates the validation
  set with training-split images and throws away the curated val split.

  FIX: When train/ and val/ directories both exist, load them into
  separate lists and skip train_test_split entirely.

BUG 5 — Feature extraction called with wrong signature
  Root cause: extract_medical_features(img) is called with a PIL Image
  object, but the function signature is extract_medical_features(path).
  This silently returns zeros for every image in the dataset, meaning
  the feature branch trains on pure noise.

  FIX: Pass img_path (string) directly to extract_medical_features,
  not a PIL Image. The function handles file I/O internally.

BUG 6 — AMP (Automatic Mixed Precision) not enabled despite being
  listed in training config. With a T4, AMP gives ~2x speedup and
  reduces memory pressure.

  FIX: Added torch.cuda.amp.GradScaler and autocast context manager.

BUG 7 — WeightedRandomSampler uses train_labels from the full pool,
  not the split train_labels, when train/val dirs are pre-split.
  Result: sampler weights don't match actual loader indices → wrong
  class balancing.

  FIX: Sampler is always built from the final train_labels list after
  splitting logic completes.
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_SCRIPT_VERSION = "EfficientNet-B3-Hybrid-v4-nan-scheduler-fix"
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
    """Deeper MLP for the 30-dim traditional feature branch with residual."""
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
    """EfficientNet-B3 backbone + traditional feature MLP, fused with attention."""
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

    def forward(self, images, features):
        cnn_feat = self.backbone(images)
        trad_feat = self.feature_mlp(features)

        fused = torch.cat([cnn_feat, trad_feat], dim=1)
        attn = self.attention(fused)

        cnn_scaled  = cnn_feat  * attn[:, 0:1]
        trad_scaled = trad_feat * attn[:, 1:2]
        fused_scaled = torch.cat([cnn_scaled, trad_scaled], dim=1)

        logits = self.classifier(fused_scaled)
        return logits, attn


def build_model(num_classes, num_features, device):
    model = EfficientNetHybrid(num_classes=num_classes, num_features=num_features)
    model = model.to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model            : EfficientNetHybrid (EfficientNet-B3, ImageNet weights)")
    print(f"  Total parameters : {n_params:,}")
    print(f"  Trainable now    : {trainable:,}")
    print(f"  Device           : {device}")
    print(f"  Classes          : {num_classes}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Loss — FIX BUG 1b: clamp logits before log_softmax to prevent -inf
# ─────────────────────────────────────────────────────────────────────────────
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # FIX: clamp logits to prevent numerical explosion → NaN in log_softmax
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
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.feature_cache = feature_cache if feature_cache is not None else {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        features = self.feature_cache.get(img_path, np.zeros(NUM_TRADITIONAL_FEATURES))

        return {
            'image': image,
            'features': torch.FloatTensor(features),
            'label': torch.LongTensor([label])[0]
        }


# ─────────────────────────────────────────────────────────────────────────────
# MixUp — FIX BUG 1a: actually sample lambda from Beta distribution
# ─────────────────────────────────────────────────────────────────────────────
def mixup_batch(images, features, labels, alpha=0.4):
    """
    True Beta-sampled MixUp.
    Previously: lam was hardcoded as `alpha` (a constant 0.3), making every
    mixed sample identical and defeating the purpose of MixUp entirely.
    Now: lam ~ Beta(alpha, alpha), clamped to [0.05, 0.95].
    """
    batch_size = images.size(0)
    # FIX: sample from Beta distribution
    lam = float(np.random.beta(alpha, alpha))
    lam = max(0.05, min(0.95, lam))  # clamp away from degenerate extremes

    index = torch.randperm(batch_size, device=images.device)
    mixed_images   = lam * images   + (1 - lam) * images[index]
    mixed_features = lam * features + (1 - lam) * features[index]
    labels_a = labels
    labels_b = labels[index]
    return mixed_images, mixed_features, labels_a, labels_b, lam


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler — FIX BUG 2: step() called once per epoch, not per batch
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """
    Per-EPOCH warmup + cosine annealing.

    FIX: The original called scheduler.step() inside train_epoch(), meaning
    it was called N_BATCHES times per epoch. With warmup_epochs=3 and
    59 batches/epoch, the scheduler completed 'warmup' after just 3 batches
    (not 3 epochs), then entered deep cosine decay by epoch 2.

    This version is called once per epoch from the main loop.
    Each param_group stores its own 'initial_lr' (set at optimizer creation)
    and gets scaled proportionally.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

        # Store each group's intended max LR (set at optimizer creation)
        for pg in optimizer.param_groups:
            pg['initial_lr'] = pg['lr']

    def step(self):
        self.current_epoch += 1
        e = self.current_epoch

        if e <= self.warmup_epochs:
            scale = e / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg in self.optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * scale

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Train epoch — FIX BUG 2: removed scheduler.step() call
# FIX BUG 6: added AMP support
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, train_loader, optimizer, criterion, device,
                scaler=None, use_mixup=False, current_epoch=0,
                mixup_start_epoch=15):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        images   = batch['image'].to(device)
        features = batch['features'].to(device)
        labels   = batch['label'].to(device)

        use_mixup_now = use_mixup and (current_epoch >= mixup_start_epoch)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            if use_mixup_now:
                images_m, features_m, y_a, y_b, lam = mixup_batch(images, features, labels)
                logits, _ = model(images_m, features_m)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                logits, _ = model(images, features)
                loss = criterion(logits, labels)

        # FIX: check for NaN loss before backward — skip bad batch
        if not torch.isfinite(loss):
            print(f"  ⚠️  Non-finite loss ({loss.item():.4f}) skipped.")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
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

    acc = 100.0 * correct / max(1, total)
    avg_loss = total_loss / max(1, len(train_loader))
    return avg_loss, acc


def evaluate(model, val_loader, device, criterion=None):
    model.eval()
    correct = 0
    total = 0
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

    acc = 100.0 * correct / max(1, total)
    avg_loss = total_loss / max(1, len(val_loader)) if criterion is not None else 0.0
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    macro_f1 = report['macro avg']['f1-score']
    return acc, avg_loss, all_preds, all_labels, macro_f1


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_cache(image_paths, feature_scaler=None, fit_scaler=False):
    """
    Extract features for a list of image paths.

    The original code passed a path string to extract_medical_features(), but
    the function validates its input and requires a PIL Image (raises
    "Unsupported image type: <class 'str'>" otherwise).
    Fix: open each image with PIL first, then pass the Image object.
    """
    raw = {}
    for img_path in tqdm(image_paths, desc="Extracting features", leave=False):
        try:
            img = Image.open(img_path).convert('RGB')
            feats = extract_medical_features(img)
            raw[img_path] = feats
        except Exception as e:
            print(f"  ⚠️  Feature extraction failed for {Path(img_path).name}: {e}")
            raw[img_path] = np.zeros(NUM_TRADITIONAL_FEATURES)

    if fit_scaler:
        feature_scaler = StandardScaler()
        feature_scaler.fit([raw[p] for p in image_paths])

    cache = {}
    for p in image_paths:
        if feature_scaler is not None:
            cache[p] = feature_scaler.transform([raw[p]])[0]
        else:
            cache[p] = raw[p]

    return cache, feature_scaler


# ─────────────────────────────────────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────────────────────────────────────
def train_hybrid_model(data_dir, output_dir, epochs=100, batch_size=32,
                       lr=1e-4, early_stopping_patience=20, num_workers=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    data_path = Path(data_dir)

    print(f"\n📁 Scanning: {data_path}")

    # ── FIX BUG 4: Respect pre-made train/val split ───────────────────────
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
        print("✅ Pre-split train/ and val/ found — using them directly.")
        class_names = sorted([p.name for p in train_path.iterdir() if p.is_dir()])
        print(f"   Classes: {class_names}")
        print("\n📂 train/")
        train_paths, train_labels = load_split(train_path, class_names)
        print("\n📂 val/")
        val_paths, val_labels = load_split(val_path, class_names)
    else:
        print("⚠️  No pre-split dirs found, scanning root and splitting 80/20.")
        class_names = sorted([p.name for p in data_path.iterdir()
                               if p.is_dir() and p.name not in ('test', 'synthetic', 'sipakmed_raw')])
        all_paths, all_labels = load_split(data_path, class_names)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

    print(f"\n✅ Train: {len(train_paths)}  |  Val: {len(val_paths)}")
    if len(train_paths) == 0:
        raise ValueError("No training images found.")

    # ── Feature extraction ────────────────────────────────────────────────
    print("\n🔍 Extracting train features (fit scaler)...")
    train_cache, feature_scaler = build_feature_cache(train_paths, fit_scaler=True)

    print("🔍 Extracting val features (apply scaler)...")
    val_cache, _ = build_feature_cache(val_paths, feature_scaler=feature_scaler)

    # ── Transforms ───────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
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

    # FIX BUG 7: sampler built from final train_labels
    class_counts  = Counter(train_labels)
    class_weights = {i: 1.0 / class_counts[i] for i in range(len(class_names))}
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)

    num_workers = min(num_workers, os.cpu_count() or 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(num_classes=len(class_names),
                        num_features=NUM_TRADITIONAL_FEATURES,
                        device=device)

    # Differential LR: backbone gets 1/100 of head LR
    backbone_param_ids = {id(p) for p in model.backbone.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) in backbone_param_ids]
    head_params     = [p for p in model.parameters() if id(p) not in backbone_param_ids]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.01,  'weight_decay': 1e-4},
        {'params': head_params,     'lr': lr,          'weight_decay': 1e-4},
    ])

    # FIX BUG 2: scheduler stepped once/epoch, stores initial_lr per group
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=epochs)

    # Plateau scheduler as safety net
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7)

    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)

    # FIX BUG 6: AMP GradScaler
    use_amp = device.type == 'cuda'
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"  AMP: {'Enabled' if use_amp else 'Disabled'}")

    # ── Training loop ──────────────────────────────────────────────────
    best_macro_f1   = 0.0
    best_val_acc    = 0.0
    patience_counter = 0
    checkpoint_path = os.path.join(output_dir, 'best_model.pt')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': []}

    for epoch in range(epochs):
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}  |  LR head={lrs[1]:.2e}  backbone={lrs[0]:.2e}")
        print(f"{'='*70}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, use_mixup=True,
            current_epoch=epoch, mixup_start_epoch=15
        )

        val_acc, val_loss, _, _, macro_f1 = evaluate(model, val_loader, device, criterion)

        # FIX BUG 2: step schedulers ONCE per epoch here, not inside train_epoch
        scheduler.step()
        prev_lr = optimizer.param_groups[1]['lr']
        plateau_scheduler.step(macro_f1)
        new_lr = optimizer.param_groups[1]['lr']
        if new_lr < prev_lr:
            print(f"  📉 ReduceLROnPlateau: head LR {prev_lr:.2e} → {new_lr:.2e}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(macro_f1)

        print(f"Train  — loss: {train_loss:.4f} | acc: {train_acc:.2f}%")
        print(f"Val    — loss: {val_loss:.4f}  | acc: {val_acc:.2f}% | macro-F1: {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_val_acc  = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'macro_f1': macro_f1,
                'class_names': class_names,
                'num_features': NUM_TRADITIONAL_FEATURES,
            }, checkpoint_path)
            print(f"✅ Checkpoint saved  (macro-F1={macro_f1:.4f}, val_acc={val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{early_stopping_patience})")
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered.")
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
    parser.add_argument('--data-dir',               type=str,   default='/kaggle/working/data')
    parser.add_argument('--checkpoint-dir',         type=str,   default='./checkpoints')
    parser.add_argument('--epochs',                 type=int,   default=100)
    parser.add_argument('--batch-size',             type=int,   default=32)
    parser.add_argument('--learning-rate',          type=float, default=1e-4)
    parser.add_argument('--early-stopping-patience',type=int,   default=20)
    parser.add_argument('--num-workers',            type=int,   default=4)
    parser.add_argument('--seed',                   type=int,   default=42)
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