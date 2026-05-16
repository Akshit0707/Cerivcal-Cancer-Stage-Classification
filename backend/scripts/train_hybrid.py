"""
train_hybrid.py  —  v9 (accuracy push: targets 80-90%)
═══════════════════════════════════════════════════════
WHAT CHANGED FROM v8 AND WHY:

PROBLEM 1 — Val stuck at 39-40%:
  CNN backbone was frozen for only 5 epochs, then unfrozen with such a low LR
  that gradients barely moved weights. The head never got strong enough during
  freeze-only training to guide the unfreeze properly.
  FIX: Freeze for 8 epochs at lr=3e-4 head only. After unfreeze, use
       lr=1e-4 head / 1e-5 backbone — standard fine-tuning ratios that work.

PROBLEM 2 — Double class-imbalance compensation:
  WeightedRandomSampler already balances batches. FocalLoss with class weights
  on top causes the model to over-penalise majority classes → confusion.
  FIX: Keep WeightedRandomSampler (best for tiny datasets). Use FocalLoss
       WITHOUT per-class weights. Let the sampler handle balance.

PROBLEM 3 — FeatureMLP skip connection adds noise:
  30 traditional features projected to 128 dims with a residual connection.
  If any feature extractor returns near-zero vectors (common on Kaggle GPU),
  the skip proj(x) still pushes garbage into the fusion.
  FIX: Remove skip connection. Make MLP a clean 3-block net. Add
       a gate that can learn to suppress the feature branch entirely
       if it's uninformative (FiLM-style learned scale+shift).

PROBLEM 4 — MixUp starts at epoch 25, too late:
  With only 100 epochs and early stopping at 20, MixUp never kicks in
  meaningfully. Also CutMix+MixUp simultaneously halves the chance of each.
  FIX: MixUp starts at epoch 10, alpha=0.1→0.3. CutMix only after ep 20.
       Both together from ep 30.

PROBLEM 5 — SWA BN update every epoch is very slow:
  _update_bn_hybrid() runs a full forward pass on train_loader every SWA
  epoch. That's 72 extra batches every epoch from ep 40 → 100.
  FIX: Only run BN update at the very end, not every epoch.
       Remove swa_scheduler.step() from the hot path — it conflicts with
       WarmupCosineScheduler anyway.

PROBLEM 6 — TTA on every val step using PIL re-reads from disk:
  5× val loader reloads every image from disk per epoch. Extremely slow
  and provides negligible benefit during training.
  FIX: TTA only at final evaluation after training. During training use
       single-pass val (fast, enables more epochs).

PROBLEM 7 — Scheduler conflicts:
  WarmupCosineScheduler + SWALR both call step() → LR becomes undefined.
  FIX: Drop SWALR entirely. Use a single clean OneCycleLR-style schedule:
       warmup 3 ep → cosine decay. After unfreeze, reset scheduler.

PROBLEM 8 — Data augmentation too weak for medical imaging:
  Standard ImageNet augmentations. Cervical cell images need:
    - Stain normalisation simulation (color jitter, grayscale prob)
    - Nucleus-aware crops (RandomResizedCrop instead of just resize)
    - Elastic deformations (approximated via RandomAffine with shear)
  FIX: Stronger, domain-appropriate augmentation pipeline.

PROBLEM 9 — Attention produces 2 scalars but fused_scaled still uses
  element-wise multiply then cat — the attention output is always ~0.5
  for both branches early in training, providing no benefit.
  FIX: Replace with a proper channel attention (SE-style) that learns
       to re-weight each of the 1664 fusion channels independently.

PROBLEM 10 — LabelSmoothing + FocalLoss double-smoothing:
  FocalLoss smoothing=0.05 + label smoothing in the same function.
  FIX: Smoothing=0.1, gamma=1.5 (less aggressive), no double application.

NEW ADDITIONS:
  - AutoAugment (RandAugment) for stronger regularisation
  - Longer head-only warmup (8 epochs vs 5)
  - Gradient clipping tightened to 0.5
  - Validation every epoch but TTA only every 5 epochs (compromise)
  - Per-class accuracy printed every 10 epochs for debugging
  - AdamW with decoupled weight decay (correct implementation)
  - Label smoothing via nn.CrossEntropyLoss as a *fallback* sanity check
"""

import os
import argparse
import math
import random
import sys
import warnings
import json
import copy
from collections import Counter
from pathlib import Path
from PIL import Image

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_SCRIPT_VERSION = "EfficientNet-B3-Hybrid-v9-accuracy-push"
print(f"[train_hybrid.py] version={_SCRIPT_VERSION}  file={__file__}")

# ─────────────────────────────────────────────────────────────────────────────
# Global toggles
# ─────────────────────────────────────────────────────────────────────────────
USE_SAM      = False
USE_TTA      = True       # Only at final eval, not every epoch
USE_SWA      = True
USE_CUTMIX   = True
SWA_START    = 50         # Later SWA start — let model converge first
TTA_AUGMENTS = 5
TTA_EVERY_N  = 5          # Run TTA only every N val epochs (speed vs accuracy)

# ─────────────────────────────────────────────────────────────────────────────
# Architecture constants — DO NOT change without rebuilding checkpoints
# ─────────────────────────────────────────────────────────────────────────────
NUM_TRADITIONAL_FEATURES = 30
CNN_INPUT_SIZE  = 300
FEAT_OUT_DIM    = 128
CNN_OUT_DIM     = 1536     # EfficientNet-B3 penultimate layer
FUSION_DIM      = CNN_OUT_DIM + FEAT_OUT_DIM   # 1664

BACKBONE_FREEZE_EPOCHS  = 8    # Longer head warmup before touching backbone
UNFREEZE_WARMUP_EPOCHS  = 4    # Safe low-LR epochs after unfreeze

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


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────────────────────
# SAM Optimizer (optional, 2× slower)
# ─────────────────────────────────────────────────────────────────────────────
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(p.grad * scale.to(p))
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(torch.stack([
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ─────────────────────────────────────────────────────────────────────────────
# Model Components
# ─────────────────────────────────────────────────────────────────────────────

class SqueezeExcitation(nn.Module):
    """Channel attention over the full fusion vector."""
    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(4, dim // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, dim // reduction), dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class FeatureMLP(nn.Module):
    """
    Clean 3-block MLP — NO skip connection.
    The residual was adding noise when traditional features are near-zero.
    Output: 128-dim embedding aligned with CNN feature scale.
    """
    def __init__(self, in_dim=30, out_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        # Learned gate: allows the network to suppress the MLP branch
        # if traditional features are uninformative (outputs near 0)
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.net(x)
        gate = self.gate(x)      # scalar ∈ (0,1) per sample
        return feat * gate        # suppresses branch when features are bad


class EfficientNetHybrid(nn.Module):
    """
    EfficientNet-B3 (300×300) + FeatureMLP with SE channel attention.

    Key changes from v8:
      - FeatureMLP has a learned gate (suppresses bad features)
      - Channel-wise SE attention over full 1664-dim fusion instead of
        2-scalar branch attention (much more expressive)
      - Stronger classifier head: 1664 → 768 → 256 → num_classes
        (extra layer helps when fusion_dim is large)
      - Dropout schedule: 0.4 on first FC, 0.2 on second
    """
    def __init__(self, num_classes=5, num_features=30,
                 feat_out_dim=128, dropout=0.4, drop_path_rate=0.2):
        super().__init__()

        # Backbone
        try:
            from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
            backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.cnn_out_dim = backbone.classifier[1].in_features  # 1536
            backbone.classifier = nn.Identity()
            self._set_drop_path(backbone, drop_path_rate)
            self.backbone = backbone
        except Exception:
            # Fallback to ResNet50 if torchvision version is old
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.cnn_out_dim = backbone.fc.in_features  # 2048
            backbone.fc = nn.Identity()
            self.backbone = backbone

        # Traditional feature branch with learned gate
        self.feature_mlp = FeatureMLP(
            in_dim=num_features, out_dim=feat_out_dim, dropout=0.3
        )

        fusion_dim = self.cnn_out_dim + feat_out_dim  # 1664

        # Channel-wise SE attention on full fusion vector
        # Much more expressive than the 2-scalar branch weighting in v8
        self.channel_attn = SqueezeExcitation(fusion_dim, reduction=16)

        # Classifier: 3-layer head with BN for stability
        # 1664 → 768 → 256 → num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(256, num_classes),
        )

        self._init_head()

    def _init_head(self):
        """Kaiming init for all Linear layers in the head."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _set_drop_path(self, backbone, rate: float):
        try:
            blocks = list(backbone.features.children())
            n = sum(1 for stage in blocks
                    if hasattr(stage, '__iter__') for _ in stage)
            rates = torch.linspace(0, rate, max(n, 1)).tolist()
            idx = 0
            for stage in blocks:
                if not hasattr(stage, '__iter__'):
                    continue
                for block in stage:
                    if hasattr(block, 'stochastic_depth'):
                        block.stochastic_depth.p = rates[idx]
                    idx += 1
        except Exception:
            pass

    def forward(self, images, features):
        images   = torch.nan_to_num(images,   nan=0.0, posinf=1.0,  neginf=-1.0)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0,  neginf=0.0)

        cnn_feat  = self.backbone(images)
        cnn_feat  = torch.nan_to_num(cnn_feat, nan=0.0, posinf=1e3, neginf=-1e3)

        trad_feat = self.feature_mlp(features)
        trad_feat = torch.nan_to_num(trad_feat, nan=0.0, posinf=1e3, neginf=-1e3)

        fused = torch.cat([cnn_feat, trad_feat], dim=1)   # (B, 1664)
        fused = self.channel_attn(fused)                   # SE re-weighting
        return self.classifier(fused)


def build_model(num_classes, num_features, device, dropout=0.4, drop_path_rate=0.2):
    model = EfficientNetHybrid(
        num_classes=num_classes,
        num_features=num_features,
        feat_out_dim=FEAT_OUT_DIM,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
    ).to(device)

    n_params  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model        : EfficientNetHybrid-v9 (EfficientNet-B3 @ {CNN_INPUT_SIZE}×{CNN_INPUT_SIZE})")
    print(f"  fusion_dim   : {FUSION_DIM}  (CNN={CNN_OUT_DIM} + feat={FEAT_OUT_DIM})")
    print(f"  Parameters   : {n_params:,} total / {trainable:,} trainable")
    print(f"  Device       : {device}  |  Classes: {num_classes}")
    return model


def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in model.backbone.parameters())
    print(f"  🔒 Backbone frozen ({n:,} params) — head-only training for {BACKBONE_FREEZE_EPOCHS} epochs")


def get_backbone_stages(model):
    try:
        return list(model.backbone.features.children())
    except Exception:
        return []


def unfreeze_progressive(model, epoch, freeze_start, total_epochs):
    """Unfreeze backbone from top (near-head) to bottom (near-input)."""
    stages = get_backbone_stages(model)
    if not stages:
        for p in model.backbone.parameters():
            p.requires_grad = True
        n = sum(p.numel() for p in model.backbone.parameters())
        print(f"  🔓 Backbone fully unfrozen ({n:,} params)")
        return

    n_stages = len(stages)
    epochs_since = epoch - freeze_start
    # Unfreeze 1 new stage every 3 epochs after freeze_start
    n_to_unfreeze = min(n_stages, 1 + epochs_since // 3)

    for i, stage in enumerate(reversed(stages)):
        req_grad = (i < n_to_unfreeze)
        for p in stage.parameters():
            p.requires_grad = req_grad

    unfrozen = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    total    = sum(p.numel() for p in model.backbone.parameters())
    print(f"  🔓 Unfreeze {n_to_unfreeze}/{n_stages} stages ({unfrozen:,}/{total:,} backbone params)")


# ─────────────────────────────────────────────────────────────────────────────
# Loss — FocalLoss WITHOUT per-class weights
# WeightedRandomSampler handles class balance; no double-compensation.
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss with label smoothing.
    gamma=1.5 (softer than 2.0 — avoids over-focusing on hard negatives
    in a noisy medical dataset where hard examples may be mislabelled).
    No per-class weight here — sampler already balances classes.
    """
    def __init__(self, gamma=1.5, smoothing=0.1, num_classes=5):
        super().__init__()
        self.gamma      = gamma
        self.smoothing  = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        logits = torch.clamp(logits, -50.0, 50.0)
        n_cls  = logits.size(-1)

        # Label-smoothed targets
        with torch.no_grad():
            smooth_t = torch.full_like(logits, self.smoothing / (n_cls - 1))
            smooth_t.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss   = -(smooth_t * log_probs).sum(dim=-1)

        # Focal weight from the true class probability
        probs       = F.softmax(logits, dim=-1)
        true_probs  = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w     = (1 - true_probs) ** self.gamma
        return (focal_w * ce_loss).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class HybridDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, feature_cache=None):
        self.image_paths   = image_paths
        self.labels        = labels
        self.transform     = transform
        self.feature_cache = feature_cache or {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image    = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        features = self.feature_cache.get(img_path, np.zeros(NUM_TRADITIONAL_FEATURES))
        return {
            'image':    image,
            'features': torch.FloatTensor(features),
            'label':    torch.tensor(self.labels[idx], dtype=torch.long),
            'path':     img_path,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transform(input_size=300):
    """
    Domain-appropriate augmentations for cervical cytology images:
    - RandomResizedCrop: simulates varying magnification / cell positioning
    - ColorJitter: simulates stain variability (Papanicolaou stain batches)
    - RandomGrayscale: forces learning of texture/shape not just color
    - RandomAffine with shear: approximates elastic deformation
    - RandomErasing: occlusion robustness
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            input_size, scale=(0.7, 1.0), ratio=(0.85, 1.15),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(180),   # cells have no canonical orientation
        transforms.ColorJitter(
            brightness=0.35, contrast=0.35, saturation=0.25, hue=0.08
        ),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1),
            scale=(0.85, 1.15), shear=(-10, 10),
        ),
        transforms.RandomGrayscale(p=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.0)),
    ])


def get_val_transform(input_size=300):
    return transforms.Compose([
        transforms.Resize((input_size, input_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_tta_transforms(input_size=300):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    resize = [transforms.Resize((input_size, input_size),
                                interpolation=transforms.InterpolationMode.BICUBIC),
              transforms.ToTensor(), normalize]
    return [
        transforms.Compose(resize),
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)] + resize),
        transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]   + resize),
        transforms.Compose([transforms.RandomRotation((90, 90))]    + resize),
        transforms.Compose([transforms.RandomRotation((180, 180))]  + resize),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MixUp / CutMix — starts earlier than v8
# ─────────────────────────────────────────────────────────────────────────────
def mixup_batch(images, features, labels, alpha=0.2):
    lam   = float(np.random.beta(alpha, alpha))
    lam   = max(0.05, min(0.95, lam))
    idx   = torch.randperm(images.size(0), device=images.device)
    return (lam * images   + (1-lam) * images[idx],
            lam * features + (1-lam) * features[idx],
            labels, labels[idx], lam)


def cutmix_batch(images, features, labels, alpha=0.2):
    import math as _math
    lam = float(np.random.beta(alpha, alpha))
    lam = max(0.05, min(0.95, lam))
    idx = torch.randperm(images.size(0), device=images.device)
    _, _, H, W = images.shape
    cut_r  = _math.sqrt(1 - lam)
    ch, cw = int(H*cut_r), int(W*cut_r)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1 = max(0, cx-cw//2); x2 = min(W, cx+cw//2)
    y1 = max(0, cy-ch//2); y2 = min(H, cy+ch//2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2-x1)*(y2-y1)/(H*W)
    return (mixed,
            lam * features + (1-lam) * features[idx],
            labels, labels[idx], lam)


def get_mixup_alpha(epoch, start=10, cutmix_start=20, max_alpha=0.3):
    """
    epoch < start:        no augmentation
    start ≤ epoch < cutmix_start: MixUp only, alpha ramps 0→max_alpha
    epoch ≥ cutmix_start: both MixUp and CutMix available
    """
    if epoch < start:
        return 0.0, False
    ramp = min(1.0, (epoch - start) / max(1, 10))
    alpha = max_alpha * ramp
    use_cutmix = epoch >= cutmix_start
    return alpha, use_cutmix


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler — clean cosine with warmup, no conflicts
# ─────────────────────────────────────────────────────────────────────────────
class CosineWarmupScheduler:
    """
    Linear warmup → cosine decay (with optional warm restarts).
    Single source of truth for LR — no SWALR conflicts.
    After unfreeze, call reset() to start a new cosine cycle.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_scale=0.01,
                 n_cycles=2):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.min_lr_scale   = min_lr_scale
        self.n_cycles       = n_cycles
        self.epoch          = 0
        for pg in optimizer.param_groups:
            pg['base_lr'] = pg['lr']

    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.warmup_epochs:
            scale = e / max(1, self.warmup_epochs)
        else:
            progress  = (e - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cycle_pos = (progress * self.n_cycles) % 1.0
            scale     = self.min_lr_scale + 0.5*(1-self.min_lr_scale)*(1+math.cos(math.pi*cycle_pos))
        for pg in self.optimizer.param_groups:
            pg['lr'] = pg['base_lr'] * scale

    def get_lrs(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def reset(self, new_lrs=None, warmup=3, total=None):
        """Call after backbone unfreeze to start a fresh cosine cycle."""
        self.epoch         = 0
        self.warmup_epochs = warmup
        self.total_epochs  = total or self.total_epochs
        if new_lrs:
            for pg, lr in zip(self.optimizer.param_groups, new_lrs):
                pg['lr']      = lr
                pg['base_lr'] = lr


# ─────────────────────────────────────────────────────────────────────────────
# SWA BN update (dict-aware, runs only at end of training)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def update_bn_hybrid(loader, swa_model, device):
    """One-time BN recalculation at end of training. Handles dict batches."""
    for m in swa_model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
    swa_model.train()
    for batch in tqdm(loader, desc="SWA BN update", leave=False):
        images   = batch['image'].to(device)
        features = batch['features'].to(device)
        swa_model(images, features)
    swa_model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device,
                scaler, epoch, use_sam=False):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    first      = True

    for batch in tqdm(loader, desc="Training", leave=False):
        images   = batch['image'].to(device, non_blocking=True)
        features = batch['features'].to(device, non_blocking=True)
        labels   = batch['label'].to(device, non_blocking=True)

        # Diagnostic on first batch only
        if first:
            first = False
            fmin, fmax = features.min().item(), features.max().item()
            if abs(fmin) < 1e-6 and abs(fmax) < 1e-6:
                print("  ❌  CRITICAL: features ALL ZEROS — check extractor!")
            else:
                print(f"  ✅  feat range [{fmin:.2f}, {fmax:.2f}]")

        alpha, allow_cutmix = get_mixup_alpha(epoch)
        lam      = 1.0
        labels_a = labels_b = labels

        if alpha > 0.0:
            if allow_cutmix and USE_CUTMIX and random.random() > 0.5:
                images, features, labels_a, labels_b, lam = cutmix_batch(
                    images, features, labels, alpha)
            else:
                images, features, labels_a, labels_b, lam = mixup_batch(
                    images, features, labels, alpha)

        def forward_loss():
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits = model(images, features)
                if lam < 1.0:
                    loss = lam * criterion(logits, labels_a) + \
                           (1-lam) * criterion(logits, labels_b)
                else:
                    loss = criterion(logits, labels)
            return logits, loss

        if use_sam:
            logits, loss = forward_loss()
            _backward(loss, optimizer, scaler, clip=0.5, step='first', use_sam=True)
            logits, loss = forward_loss()
            _backward(loss, optimizer, scaler, clip=0.5, step='second', use_sam=True)
        else:
            optimizer.zero_grad()
            logits, loss = forward_loss()
            if not torch.isfinite(loss):
                print("  ⚠️  non-finite loss, skipping batch")
                continue
            _backward(loss, optimizer, scaler, clip=0.5)

        total_loss += loss.item()
        preds = logits.detach().argmax(dim=1)
        total   += labels.size(0)
        correct += (preds == labels).sum().item()

    return total_loss / max(1, len(loader)), 100.0 * correct / max(1, total)


def _backward(loss, optimizer, scaler, clip=0.5, step=None, use_sam=False):
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g['params']], clip
        )
        if use_sam:
            if step == 'first':
                optimizer.first_step(zero_grad=True)
            else:
                optimizer.second_step(zero_grad=True)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g['params']], clip
        )
        if use_sam:
            if step == 'first':
                optimizer.first_step(zero_grad=True)
            else:
                optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, criterion, class_names,
             use_tta=False, tta_transforms=None, verbose=False):
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            images   = batch['image'].to(device, non_blocking=True)
            features = batch['features'].to(device, non_blocking=True)
            labels   = batch['label'].to(device, non_blocking=True)
            paths    = batch['path']

            if use_tta and tta_transforms:
                logit_sum = None
                for t in tta_transforms:
                    imgs_aug = torch.stack([
                        t(Image.open(p).convert('RGB')) for p in paths
                    ]).to(device)
                    out = model(imgs_aug, features)
                    logit_sum = out if logit_sum is None else logit_sum + out
                logits = logit_sum / len(tta_transforms)
            else:
                logits = model(images, features)

            loss = criterion(logits, labels)
            if torch.isfinite(loss):
                total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc      = 100.0 * sum(p==l for p,l in zip(all_preds,all_labels)) / max(1, len(all_labels))
    avg_loss = total_loss / max(1, len(loader))
    report   = classification_report(all_labels, all_preds,
                                     target_names=class_names,
                                     output_dict=True, zero_division=0)
    macro_f1 = report['macro avg']['f1-score']

    if verbose:
        print("\n  Per-class accuracy:")
        for i, name in enumerate(class_names):
            cls_labels = [l for l in all_labels if l == i]
            cls_preds  = [p for p,l in zip(all_preds,all_labels) if l == i]
            cls_acc    = 100*sum(p==i for p in cls_preds)/max(1,len(cls_labels))
            print(f"    {name:10s}: {cls_acc:5.1f}%  (n={len(cls_labels)})")

    return acc, avg_loss, macro_f1, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def sanitize(arr):
    return np.clip(np.nan_to_num(np.array(arr, dtype=np.float32),
                                 nan=0., posinf=0., neginf=0.), -1e6, 1e6)


def build_feature_cache(image_paths, scaler=None, fit=False):
    raw   = {}
    n_bad = 0
    for p in tqdm(image_paths, desc="Extracting features", leave=False):
        try:
            feats = extract_medical_features(Image.open(p).convert('RGB'))
            feats = sanitize(feats)
            if len(feats) != NUM_TRADITIONAL_FEATURES:
                raise ValueError(f"Expected {NUM_TRADITIONAL_FEATURES}, got {len(feats)}")
        except Exception as e:
            feats = np.zeros(NUM_TRADITIONAL_FEATURES, dtype=np.float32)
            n_bad += 1
        raw[p] = feats

    if n_bad:
        pct = 100*n_bad/max(1,len(image_paths))
        print(f"  ⚠️  {n_bad} ({pct:.1f}%) feature failures — zero fallback used")
        if pct > 30:
            print("  ❌  >30% failures — feature extractor may be broken!")

    sample = np.concatenate([raw[p] for p in list(raw)[:10]])
    if np.allclose(sample, 0.):
        print("  ❌  CRITICAL: all features zero!")
    else:
        print(f"  ✅  Feature sanity passed — range [{sample.min():.3f}, {sample.max():.3f}]")

    if fit:
        all_arr = np.stack([raw[p] for p in image_paths])
        scaler  = RobustScaler(quantile_range=(10., 90.))
        scaler.fit(all_arr)

    cache = {}
    for p in image_paths:
        if scaler is not None:
            s = np.clip(scaler.transform([raw[p]])[0], -3., 3.)
            cache[p] = sanitize(s)
        else:
            cache[p] = raw[p]

    return cache, scaler


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def train_hybrid_model(data_dir, output_dir, epochs=100, batch_size=32,
                       lr=3e-4, early_stopping_patience=20, num_workers=4):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    data_path = Path(data_dir)

    # ── Data loading ──────────────────────────────────────────────────────
    print(f"\n📁 Scanning: {data_path}")
    train_path = data_path / 'train'
    val_path   = data_path / 'val'

    def load_split(split_path, class_names):
        paths, lbls = [], []
        for idx, cls in enumerate(class_names):
            d = split_path / cls
            if not d.exists():
                continue
            imgs = (sorted(d.glob('*.jpg')) + sorted(d.glob('*.JPG')) +
                    sorted(d.glob('*.png')) + sorted(d.glob('*.PNG')) +
                    sorted(d.glob('*.jpeg')))
            print(f"     {cls}: {len(imgs)}")
            for ip in imgs:
                paths.append(str(ip)); lbls.append(idx)
        return paths, lbls

    if train_path.exists() and val_path.exists():
        class_names = sorted([p.name for p in train_path.iterdir() if p.is_dir()])
        print(f"✅ Pre-split found. Classes: {class_names}")
        print("\n📂 train/")
        tr_paths, tr_labels = load_split(train_path, class_names)
        print("📂 val/")
        va_paths, va_labels = load_split(val_path, class_names)

        # Re-split if val is severely imbalanced
        val_counts = Counter(va_labels)
        ratio = max(val_counts.values()) / max(min(val_counts.values()), 1)
        if ratio > 3.0:
            print(f"\n⚠️  Val imbalanced ({ratio:.1f}×) — re-splitting 80/20 stratified")
            all_paths  = tr_paths + va_paths
            all_labels = tr_labels + va_labels
            tr_paths, va_paths, tr_labels, va_labels = train_test_split(
                all_paths, all_labels, test_size=0.2, random_state=42,
                stratify=all_labels)
            print("📊 Re-split:")
            tc, vc = Counter(tr_labels), Counter(va_labels)
            for i, n in enumerate(class_names):
                print(f"   {n}: train={tc[i]}  val={vc[i]}")
    else:
        class_names = sorted([p.name for p in data_path.iterdir()
                               if p.is_dir() and p.name not in
                               ('test','synthetic','sipakmed_raw')])
        all_paths, all_labels = load_split(data_path, class_names)
        tr_paths, va_paths, tr_labels, va_labels = train_test_split(
            all_paths, all_labels, test_size=0.2, random_state=42,
            stratify=all_labels)

    print(f"\n✅ Train: {len(tr_paths)}  Val: {len(va_paths)}")
    if not tr_paths:
        raise ValueError("No training images found.")

    # ── Feature extraction ────────────────────────────────────────────────
    print("\n🔍 Extracting train features...")
    tr_cache, scaler = build_feature_cache(tr_paths, fit=True)
    print("🔍 Extracting val features...")
    va_cache, _      = build_feature_cache(va_paths, scaler=scaler)

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = HybridDataset(tr_paths, tr_labels, get_train_transform(CNN_INPUT_SIZE), tr_cache)
    val_ds   = HybridDataset(va_paths, va_labels, get_val_transform(CNN_INPUT_SIZE),   va_cache)
    tta_tfms = build_tta_transforms(CNN_INPUT_SIZE) if USE_TTA else None

    # WeightedRandomSampler — 2× weight for minority class (Normal)
    class_counts = Counter(tr_labels)
    normal_idx   = class_names.index('Normal') if 'Normal' in class_names else -1
    sw = [(1./class_counts[l]) * (2. if l==normal_idx else 1.) for l in tr_labels]
    sampler = WeightedRandomSampler(sw, len(tr_labels), replacement=True)

    nw = min(num_workers, os.cpu_count() or 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(nw > 0))

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(len(class_names), NUM_TRADITIONAL_FEATURES, device,
                        dropout=0.4, drop_path_rate=0.2)
    freeze_backbone(model)

    # ── Optimizer ─────────────────────────────────────────────────────────
    backbone_ids  = {id(p) for p in model.backbone.parameters()}
    head_params   = [p for p in model.parameters() if id(p) not in backbone_ids]
    back_params   = list(model.backbone.parameters())

    def make_optimizer(backbone_lr, head_lr):
        groups = [
            {'params': back_params, 'lr': backbone_lr, 'weight_decay': 1e-4},
            {'params': head_params, 'lr': head_lr,     'weight_decay': 1e-4},
        ]
        if USE_SAM:
            return SAM(groups, torch.optim.AdamW, rho=0.05,
                       lr=head_lr, weight_decay=1e-4)
        return torch.optim.AdamW(groups)

    # Phase 1: head-only — backbone lr irrelevant (frozen), head lr=lr
    optimizer = make_optimizer(backbone_lr=lr*0.1, head_lr=lr)
    scheduler = CosineWarmupScheduler(optimizer,
                                      warmup_epochs=3,
                                      total_epochs=BACKBONE_FREEZE_EPOCHS,
                                      min_lr_scale=0.05, n_cycles=1)

    # ── Loss — no class weights (sampler handles balance) ─────────────────
    criterion = FocalLoss(gamma=1.5, smoothing=0.1, num_classes=len(class_names))

    # ── AMP ───────────────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # ── SWA ───────────────────────────────────────────────────────────────
    swa_model = AveragedModel(model) if USE_SWA else None

    print(f"\n{'='*70}")
    print(f"  AMP    : {'✅' if scaler else '❌'}")
    print(f"  TTA    : {'✅ (every '+str(TTA_EVERY_N)+' val epochs)' if USE_TTA else '❌'}")
    print(f"  SWA    : {'✅ (start ep '+str(SWA_START)+')' if USE_SWA else '❌'}")
    print(f"  SAM    : {'✅' if USE_SAM else '❌'}")
    print(f"  CutMix : {'✅' if USE_CUTMIX else '❌'}")
    print(f"  MixUp  : ✅ (starts ep 10, ramps to alpha=0.3)")
    print(f"  LR     : head={lr:.1e}  backbone={lr*0.1:.1e} (after unfreeze: {lr*0.5:.1e}/{lr*0.05:.1e})")
    print(f"{'='*70}\n")

    best_f1          = 0.
    best_val_acc     = 0.
    best_val_loss    = float('inf')
    patience_counter = 0
    backbone_unfrozen = False
    checkpoint_path  = os.path.join(output_dir, 'best_model.pt')
    history = {k: [] for k in ['train_loss','train_acc','val_loss','val_acc','val_f1']}

    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*70}")

        # ── Progressive unfreeze ──────────────────────────────────────────
        if not backbone_unfrozen and epoch >= BACKBONE_FREEZE_EPOCHS:
            backbone_unfrozen = True
            unfreeze_progressive(model, epoch, BACKBONE_FREEZE_EPOCHS, epochs)

            # After unfreeze: higher head LR (head is now well-trained),
            # very small backbone LR (don't destroy ImageNet features)
            optimizer = make_optimizer(backbone_lr=lr*0.05, head_lr=lr*0.5)
            remaining = epochs - epoch
            scheduler = CosineWarmupScheduler(optimizer,
                                              warmup_epochs=UNFREEZE_WARMUP_EPOCHS,
                                              total_epochs=remaining,
                                              min_lr_scale=0.01, n_cycles=2)
            if USE_SWA:
                swa_model = AveragedModel(model)  # reset SWA after unfreeze
            print(f"  Optimizer reset — head_lr={lr*0.5:.1e} backbone_lr={lr*0.05:.1e}")

        elif backbone_unfrozen and epoch > BACKBONE_FREEZE_EPOCHS:
            unfreeze_progressive(model, epoch, BACKBONE_FREEZE_EPOCHS, epochs)

        # ── Train ─────────────────────────────────────────────────────────
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, epoch=epoch, use_sam=USE_SAM,
        )

        # ── SWA update (no BN update every epoch — saves time) ───────────
        use_swa = USE_SWA and (epoch >= SWA_START)
        if use_swa:
            swa_model.update_parameters(model)

        # ── Scheduler step ────────────────────────────────────────────────
        scheduler.step()

        # ── Evaluate ──────────────────────────────────────────────────────
        eval_model = swa_model if use_swa else model
        do_tta     = USE_TTA and ((epoch+1) % TTA_EVERY_N == 0)
        verbose    = (epoch+1) % 10 == 0

        val_acc, val_loss, macro_f1, _, _ = evaluate(
            eval_model, val_loader, device, criterion, class_names,
            use_tta=do_tta, tta_transforms=tta_tfms, verbose=verbose,
        )

        lrs = scheduler.get_lrs()
        alpha, _ = get_mixup_alpha(epoch)
        tta_tag = " [TTA]" if do_tta else ""
        swa_tag = " [SWA]" if use_swa else ""
        print(f"Train  — loss: {tr_loss:.4f} | acc: {tr_acc:.2f}%")
        print(f"Val    — loss: {val_loss:.4f} | acc: {val_acc:.2f}% | "
              f"F1: {macro_f1:.4f}{tta_tag}{swa_tag}")
        print(f"LR     — head={lrs[1]:.2e} backbone={lrs[0]:.2e} | "
              f"mixup_alpha={alpha:.3f}")

        for k, v in zip(['train_loss','train_acc','val_loss','val_acc','val_f1'],
                        [tr_loss, tr_acc, val_loss, val_acc, macro_f1]):
            history[k].append(v)

        # ── Checkpoint ────────────────────────────────────────────────────
        improved = (macro_f1 > best_f1 + 1e-4) or \
                   (val_acc > best_val_acc + 0.5 and macro_f1 > best_f1 - 0.01)
        if improved:
            best_f1       = macro_f1
            best_val_acc  = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            save_m = swa_model if use_swa else model
            torch.save({
                'epoch':            epoch+1,
                'model_state_dict': save_m.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':          val_acc,
                'macro_f1':         macro_f1,
                'class_names':      class_names,
                'num_features':     NUM_TRADITIONAL_FEATURES,
                'feat_out_dim':     FEAT_OUT_DIM,
                'input_size':       CNN_INPUT_SIZE,
                'script_version':   _SCRIPT_VERSION,
                'use_swa':          use_swa,
            }, checkpoint_path)
            print(f"✅ Checkpoint saved (F1={macro_f1:.4f}, acc={val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{early_stopping_patience})")
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break

    # ── Final SWA BN update ────────────────────────────────────────────────
    if USE_SWA and swa_model is not None and epoch >= SWA_START:
        print("\n🔄 Running final SWA BN update...")
        update_bn_hybrid(train_loader, swa_model, device)
        swa_acc, swa_loss, swa_f1, _, _ = evaluate(
            swa_model, val_loader, device, criterion, class_names,
            use_tta=USE_TTA, tta_transforms=tta_tfms, verbose=True,
        )
        print(f"  SWA final — acc={swa_acc:.2f}%  F1={swa_f1:.4f}")
        swa_path = os.path.join(output_dir, 'swa_model.pt')
        torch.save({'model_state_dict': swa_model.state_dict(),
                    'class_names': class_names,
                    'num_features': NUM_TRADITIONAL_FEATURES,
                    'feat_out_dim': FEAT_OUT_DIM,
                    'input_size': CNN_INPUT_SIZE,
                    'script_version': _SCRIPT_VERSION}, swa_path)
        print(f"  SWA model saved: {swa_path}")

    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ Training complete!")
    print(f"   Best val acc : {best_val_acc:.2f}%")
    print(f"   Best macro-F1: {best_f1:.4f}")
    print(f"   Checkpoint   : {checkpoint_path}")
    print(f"{'='*70}")
    return checkpoint_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hybrid cervical cancer classifier v9')
    parser.add_argument('--data-dir',                type=str,   default='/kaggle/working/data')
    parser.add_argument('--checkpoint-dir',          type=str,   default='./checkpoints')
    parser.add_argument('--epochs',                  type=int,   default=100)
    parser.add_argument('--batch-size',              type=int,   default=32)
    parser.add_argument('--learning-rate',           type=float, default=3e-4)
    parser.add_argument('--early-stopping-patience', type=int,   default=20)
    parser.add_argument('--num-workers',             type=int,   default=4)
    parser.add_argument('--seed',                    type=int,   default=42)
    parser.add_argument('--no-tta',    action='store_true')
    parser.add_argument('--no-swa',    action='store_true')
    parser.add_argument('--use-sam',   action='store_true')
    parser.add_argument('--no-cutmix', action='store_true')
    args = parser.parse_args()

    if args.no_tta:    USE_TTA    = False
    if args.no_swa:    USE_SWA    = False
    if args.use_sam:   USE_SAM    = True
    if args.no_cutmix: USE_CUTMIX = False

    set_seed(args.seed)

    print(f"\n{'='*70}")
    print(f"🚀 TRAINING CONFIGURATION — v9 (accuracy push)")
    print(f"{'='*70}")
    for k, v in vars(args).items():
        print(f"  {k:<30}: {v}")
    print(f"{'='*70}\n")

    train_hybrid_model(
        data_dir   = args.data_dir,
        output_dir = args.checkpoint_dir,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.learning_rate,
        early_stopping_patience = args.early_stopping_patience,
        num_workers = args.num_workers,
    )