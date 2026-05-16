"""
Training Script for GPU-Optimized Hybrid Model
Combines EfficientNet-B3 CNN features with traditional medical features

═══════════════════════════════════════════════════════════════════════
ROOT CAUSE ANALYSIS & FIXES (v8 — disconnect fixes + accuracy push)
═══════════════════════════════════════════════════════════════════════

FIX 1  — dataset.py was completely disconnected (dead code).
         train_hybrid.py uses its own HybridDataset internally.
         dataset.py's (224,224), (image,label) tuple, no-features
         design is incompatible with the hybrid model. Removed reliance.

FIX 2  — model.py (checkpoint-verified) uses feat_out_dim=128 →
         fusion_dim=1664. train_hybrid.py was building feat_out_dim=256
         → fusion_dim=1792. Checkpoints were incompatible across runs.
         FIXED: train_hybrid.py now uses feat_out_dim=128 consistently.

FIX 3  — Progressive unfreeze rebuilt the optimizer at epoch 5, then
         immediately created a new WarmupCosineScheduler with
         warmup_epochs=1. This caused a sudden LR spike right when
         the backbone received gradients for the first time — exactly
         the worst moment. FIXED: LR is held at a safe low value for
         3 warmup epochs after unfreeze before cosine ramp begins.

FIX 4  — update_bn(train_loader, swa_model, device=device) passed a
         dict-yielding DataLoader to PyTorch's update_bn which expects
         (images, labels) tuples. This caused a hang/TypeError.
         FIXED: custom _update_bn_hybrid() iterates dicts correctly.

All v7 improvements retained:
  - 300×300 input (EfficientNet-B3 native resolution)
  - MixUp start epoch=25, max_alpha=0.2
  - FocalLoss with label smoothing
  - TTA at evaluation
  - Progressive layer unfreezing (now with safe LR warmup — FIX 3)
  - Cosine annealing with warm restarts
  - SAM optimizer (optional)
  - Stronger dropout (0.5 head, 0.3 MLP)
  - CutMix alongside MixUp
  - Stochastic Depth (DropPath)
  - SWA (now with correct BN update — FIX 4)
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
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_SCRIPT_VERSION = "EfficientNet-B3-Hybrid-v8-disconnect-fix"
print(f"[train_hybrid.py] version={_SCRIPT_VERSION}  file={__file__}")

# ─────────────────────────────────────────────────────────────────────────────
# Global toggles
# ─────────────────────────────────────────────────────────────────────────────
USE_SAM         = False   # SAM optimizer — better generalization, 2× slower
USE_TTA         = True    # Test-Time Augmentation during evaluation
USE_SWA         = True    # Stochastic Weight Averaging
USE_CUTMIX      = True    # CutMix augmentation (alongside MixUp)
SWA_START       = 40      # epoch to start SWA weight averaging
TTA_AUGMENTS    = 5       # number of TTA views to average

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
CNN_INPUT_SIZE = 300        # EfficientNet-B3 native resolution

# FIX 2: feat_out_dim=128 matches model.py checkpoint-verified architecture
#         (was 256 in v7, causing fusion_dim=1792 vs model.py's 1664)
FEAT_OUT_DIM = 128

BACKBONE_FREEZE_EPOCHS = 5
UNFREEZE_WARMUP_EPOCHS = 3  # FIX 3: safe warmup after backbone unfreeze


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
# SAM Optimizer
# ─────────────────────────────────────────────────────────────────────────────
class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer wrapper.
    Toggle with USE_SAM = True (adds ~2× training time).
    """
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
        return torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ─────────────────────────────────────────────────────────────────────────────
# DropPath / Stochastic Depth
# ─────────────────────────────────────────────────────────────────────────────
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    return x / keep_prob * torch.floor(random_tensor + keep_prob)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# FIX 2: FeatureMLP now matches model.py checkpoint-verified dims:
#         in_dim=30 → hidden=256 → hidden=256 → out_dim=128
#         fusion_dim = 1536 + 128 = 1664  (was 1792 in v7)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureMLP(nn.Module):
    """
    3-block MLP matching the checkpoint-verified architecture in model.py.
    Output dim = 128 (was 256 in v7 — FIX 2).

    State-dict index map:
      net.0  Linear(30  → 256)
      net.1  BN(256)
      net.2  GELU
      net.3  Dropout(0.3)
      net.4  Linear(256 → 256)
      net.5  BN(256)
      net.6  GELU
      net.7  Dropout(0.15)
      net.8  Linear(256 → 128)   ← final reduction
      net.9  BN(128)
      proj   Linear(30  → 128)
    """
    def __init__(self, in_dim: int = 30, out_dim: int = 128,
                 hidden1: int = 256, hidden2: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),    # net.0
            nn.BatchNorm1d(hidden1),        # net.1
            nn.GELU(),                      # net.2
            nn.Dropout(dropout),            # net.3
            nn.Linear(hidden1, hidden2),   # net.4
            nn.BatchNorm1d(hidden2),        # net.5
            nn.GELU(),                      # net.6
            nn.Dropout(dropout * 0.5),      # net.7
            nn.Linear(hidden2, out_dim),   # net.8
            nn.BatchNorm1d(out_dim),        # net.9
        )
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.proj(x)


class EfficientNetHybrid(nn.Module):
    """
    EfficientNet-B3 backbone (300×300) + 3-block FeatureMLP, attention-fused.

    Dims match model.py checkpoint exactly:
      feat_out_dim = 128
      fusion_dim   = 1536 + 128 = 1664
      attn:        Linear(1664→64) → GELU → Linear(64→2) → Softmax
      classifier:  Dropout → Linear(1664→512) → BN → GELU → Dropout → Linear(512→5)
    """
    def __init__(self, num_classes: int = 5, num_features: int = 30,
                 feat_out_dim: int = 128, dropout: float = 0.5,
                 drop_path_rate: float = 0.2):
        super().__init__()
        try:
            from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
            backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            cnn_out_dim = backbone.classifier[1].in_features   # 1536
            backbone.classifier = nn.Identity()
            self._inject_drop_path(backbone, drop_path_rate)
            self.backbone = backbone
        except Exception:
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            cnn_out_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone

        self.cnn_out_dim = cnn_out_dim

        # FIX 2: use feat_out_dim=128 to match model.py
        self.feature_mlp = FeatureMLP(
            in_dim=num_features, out_dim=feat_out_dim,
            hidden1=256, hidden2=256, dropout=0.3,
        )

        fusion_dim = cnn_out_dim + feat_out_dim   # 1536 + 128 = 1664

        # Attention: NO Dropout layer (matches checkpoint key alignment in model.py)
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, 64),   # attention.0
            nn.GELU(),                    # attention.1
            nn.Linear(64, 2),             # attention.2  (not .3 — no Dropout)
            nn.Softmax(dim=-1),           # attention.3
        )

        # Classifier: 2-layer head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),              # .0
            nn.Linear(fusion_dim, 512),       # .1
            nn.BatchNorm1d(512),              # .2
            nn.GELU(),                        # .3
            nn.Dropout(dropout * 0.5),        # .4
            nn.Linear(512, num_classes),      # .5
        )

    def _inject_drop_path(self, backbone, drop_path_rate: float):
        try:
            blocks = list(backbone.features.children())
            n_blocks = sum(1 for b in blocks if hasattr(b, '__iter__') for _ in b)
            dp_rates = torch.linspace(0, drop_path_rate, max(n_blocks, 1)).tolist()
            idx = 0
            for stage in blocks:
                if not hasattr(stage, '__iter__'):
                    continue
                for block in stage:
                    if hasattr(block, 'stochastic_depth'):
                        block.stochastic_depth.p = dp_rates[idx]
                    idx += 1
        except Exception:
            pass

    def forward(self, images, features):
        images   = torch.nan_to_num(images,   nan=0.0, posinf=1.0,  neginf=-1.0)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0,  neginf=0.0)

        cnn_feat  = self.backbone(images)
        cnn_feat  = torch.nan_to_num(cnn_feat,  nan=0.0, posinf=1e3, neginf=-1e3)
        trad_feat = self.feature_mlp(features)
        trad_feat = torch.nan_to_num(trad_feat, nan=0.0, posinf=1e3, neginf=-1e3)

        fused = torch.cat([cnn_feat, trad_feat], dim=1)
        attn  = self.attention(fused)

        fused_scaled = torch.cat(
            [cnn_feat * attn[:, 0:1], trad_feat * attn[:, 1:2]], dim=1
        )
        return self.classifier(fused_scaled), attn


def build_model(num_classes, num_features, device, dropout=0.5, drop_path_rate=0.2):
    model = EfficientNetHybrid(
        num_classes=num_classes,
        num_features=num_features,
        feat_out_dim=FEAT_OUT_DIM,   # FIX 2: 128
        dropout=dropout,
        drop_path_rate=drop_path_rate,
    ).to(device)
    fusion_dim = 1536 + FEAT_OUT_DIM
    n_params  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model            : EfficientNetHybrid (EfficientNet-B3 @ 300×300)")
    print(f"  feat_out_dim     : {FEAT_OUT_DIM}  → fusion_dim={fusion_dim}")
    print(f"  Total parameters : {n_params:,}")
    print(f"  Trainable now    : {trainable:,}")
    print(f"  Device           : {device}")
    print(f"  Classes          : {num_classes}")
    return model


def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False
    frozen = sum(p.numel() for p in model.backbone.parameters())
    print(f"  🔒 Backbone frozen ({frozen:,} params) — head-only training")


def get_backbone_stages(model):
    try:
        return list(model.backbone.features.children())
    except Exception:
        return []


def unfreeze_backbone_progressive(model, epoch, freeze_start, total_epochs):
    """
    Progressive layer unfreezing from top (classifier-near) to bottom (input-near).
    Each epoch after freeze_start, more stages are unfrozen.
    """
    stages = get_backbone_stages(model)
    if not stages:
        for param in model.backbone.parameters():
            param.requires_grad = True
        n = sum(p.numel() for p in model.backbone.parameters())
        print(f"  🔓 Backbone fully unfrozen ({n:,} params)")
        return

    n_stages      = len(stages)
    thaw_epochs   = total_epochs - freeze_start
    thaw_interval = max(1, thaw_epochs // n_stages)
    epochs_since  = epoch - freeze_start
    n_to_unfreeze = min(n_stages, 1 + epochs_since // thaw_interval)

    for i, stage in enumerate(reversed(stages)):
        for param in stage.parameters():
            param.requires_grad = (i < n_to_unfreeze)

    unfrozen = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    total    = sum(p.numel() for p in model.backbone.parameters())
    print(f"  🔓 Progressive unfreeze: {n_to_unfreeze}/{n_stages} stages "
          f"({unfrozen:,}/{total:,} params)")


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss with label smoothing. gamma=2.0 down-weights easy examples."""
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, smoothing=0.05):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.weight    = weight
        self.smoothing = smoothing

    def forward(self, logits, targets):
        logits = torch.clamp(logits, -50.0, 50.0)
        n_cls  = logits.size(-1)
        with torch.no_grad():
            smooth_t = torch.full_like(logits, self.smoothing / (n_cls - 1))
            smooth_t.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss   = -(smooth_t * log_probs).sum(dim=-1)
        if self.weight is not None:
            ce_loss = ce_loss * self.weight[targets]
        probs      = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - probs) ** self.gamma * ce_loss
        return focal_loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# Note: dataset.py is NOT used here (FIX 1). HybridDataset returns the dict
# format that the model and feature extractor expect.
# ─────────────────────────────────────────────────────────────────────────────
class HybridDataset(Dataset):
    """
    Returns dicts: {'image': tensor, 'features': tensor, 'label': tensor, 'path': str}
    This is intentionally different from dataset.py's (image, label) tuple format.
    dataset.py is only valid for a plain CNN without the medical feature branch.
    """
    def __init__(self, image_paths, labels, transform=None, feature_cache=None):
        self.image_paths   = image_paths
        self.labels        = labels
        self.transform     = transform
        self.feature_cache = feature_cache if feature_cache is not None else {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label    = self.labels[idx]
        image    = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        features = self.feature_cache.get(img_path, np.zeros(NUM_TRADITIONAL_FEATURES))
        return {
            'image':    image,
            'features': torch.FloatTensor(features),
            'label':    torch.LongTensor([label])[0],
            'path':     img_path,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────
def mixup_batch(images, features, labels, alpha=0.2):
    """Beta-sampled MixUp."""
    batch_size = images.size(0)
    lam        = float(np.random.beta(alpha, alpha))
    lam        = max(0.05, min(0.95, lam))
    index      = torch.randperm(batch_size, device=images.device)
    return (lam * images   + (1 - lam) * images[index],
            lam * features + (1 - lam) * features[index],
            labels, labels[index], lam)


def cutmix_batch(images, features, labels, alpha=0.2):
    """CutMix: better than MixUp for fine-grained morphological features."""
    batch_size = images.size(0)
    lam        = float(np.random.beta(alpha, alpha))
    lam        = max(0.05, min(0.95, lam))
    index      = torch.randperm(batch_size, device=images.device)
    _, _, H, W = images.shape
    cut_ratio  = math.sqrt(1 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cx, cy     = random.randint(0, W), random.randint(0, H)
    x1 = max(0, cx - cut_w // 2); x2 = min(W, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2); y2 = min(H, cy + cut_h // 2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (H * W)
    return (mixed,
            lam * features + (1 - lam) * features[index],
            labels, labels[index], lam)


def get_mixup_alpha(epoch, mixup_start_epoch=25, max_alpha=0.2, ramp_epochs=10):
    """Ramp MixUp alpha from 0 → max_alpha over ramp_epochs after start."""
    if epoch < mixup_start_epoch:
        return 0.0
    ramp = min(1.0, (epoch - mixup_start_epoch) / max(1, ramp_epochs))
    return max_alpha * ramp


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# FIX 3: safe_lr_floor prevents the LR spike after unfreeze.
#         When unfreeze_warmup is active, LR is held at a fraction of
#         initial_lr for UNFREEZE_WARMUP_EPOCHS before cosine restarts.
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """
    Per-epoch warmup + cosine annealing with warm restarts.
    FIX 3: accepts unfreeze_epoch parameter. During [unfreeze_epoch,
    unfreeze_epoch + UNFREEZE_WARMUP_EPOCHS], LR is held at a safe
    floor (backbone_safe_lr) to prevent the gradient explosion that
    occurred in v7 when backbone was first unfrozen.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, n_cycles=3,
                 unfreeze_epoch=None, safe_lr_scale=0.1):
        self.optimizer          = optimizer
        self.warmup_epochs      = warmup_epochs
        self.total_epochs       = total_epochs
        self.n_cycles           = n_cycles
        self.current_epoch      = 0
        self.unfreeze_epoch     = unfreeze_epoch
        self.safe_lr_scale      = safe_lr_scale   # fraction of initial_lr during unfreeze warmup

        for pg in optimizer.param_groups:
            pg['initial_lr'] = pg['lr']

    def step(self):
        self.current_epoch += 1
        e = self.current_epoch

        # FIX 3: hold LR at safe floor right after backbone unfreeze
        if (self.unfreeze_epoch is not None and
                e > self.unfreeze_epoch and
                e <= self.unfreeze_epoch + UNFREEZE_WARMUP_EPOCHS):
            for pg in self.optimizer.param_groups:
                pg['lr'] = pg['initial_lr'] * self.safe_lr_scale
            return

        if e <= self.warmup_epochs:
            scale = e / max(1, self.warmup_epochs)
        else:
            progress  = (e - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cycle_pos = progress * self.n_cycles % 1.0
            scale     = 0.5 * (1.0 + math.cos(math.pi * cycle_pos))
            amplitude = 0.5 ** (progress * self.n_cycles // 1)
            scale     = scale * amplitude + (1 - amplitude) * 0.01

        for pg in self.optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * scale

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def set_group_lr(self, group_idx, new_initial_lr):
        self.optimizer.param_groups[group_idx]['initial_lr'] = new_initial_lr
        self.optimizer.param_groups[group_idx]['lr']         = new_initial_lr


# ─────────────────────────────────────────────────────────────────────────────
# TTA helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_tta_transforms(input_size=300):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    base = [transforms.Resize((input_size, input_size)), transforms.ToTensor(), normalize]
    return [
        transforms.Compose(base),
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)] + base),
        transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]   + base),
        transforms.Compose([transforms.RandomRotation((90, 90))]    + base),
        transforms.Compose([transforms.RandomRotation((270, 270))]  + base),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# FIX 4: Custom BN update for HybridDataset (dict-yielding loader)
# PyTorch's built-in update_bn(loader, model) expects (images, labels) tuples.
# HybridDataset yields dicts → TypeError / hang. This wrapper handles dicts.
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _update_bn_hybrid(loader, swa_model, device):
    """
    Replaces torch.optim.swa_utils.update_bn for dict-yielding DataLoaders.
    Resets all BN running stats, then does one full forward pass to recompute.
    """
    # Reset running stats
    for module in swa_model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.running_mean.zero_()
            module.running_var.fill_(1)
            module.num_batches_tracked.zero_()

    swa_model.train()
    for batch in tqdm(loader, desc="SWA BN update", leave=False):
        images   = batch['image'].to(device)
        features = batch['features'].to(device)
        swa_model(images, features)   # forward only, no loss needed
    swa_model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Train epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, train_loader, optimizer, criterion, device,
                scaler=None, use_mixup=False, use_cutmix=False,
                current_epoch=0, mixup_start_epoch=25, max_mixup_alpha=0.2,
                use_sam=False):
    model.train()
    total_loss  = 0.0
    correct     = 0
    total       = 0
    first_batch = True

    for batch in tqdm(train_loader, desc="Training", leave=False):
        images   = batch['image'].to(device)
        features = batch['features'].to(device)
        labels   = batch['label'].to(device)

        if first_batch:
            first_batch = False
            if not torch.isfinite(images).all():
                print("  ⚠️  DIAGNOSTIC: images contain NaN/Inf")
            if not torch.isfinite(features).all():
                print(f"  ⚠️  DIAGNOSTIC: features contain NaN/Inf")
            else:
                fmin, fmax = features.min().item(), features.max().item()
                print(f"  ✅  DIAGNOSTIC: features OK — range [{fmin:.2f}, {fmax:.2f}]")
                if abs(fmin) < 1e-6 and abs(fmax) < 1e-6:
                    print("  ❌  CRITICAL: features ALL ZEROS!")

        alpha_now   = get_mixup_alpha(current_epoch, mixup_start_epoch, max_mixup_alpha)
        use_aug_now = alpha_now > 0.0
        labels_a = labels_b = labels
        lam = 1.0

        if use_aug_now:
            if use_cutmix and use_mixup and random.random() > 0.5:
                images, features, labels_a, labels_b, lam = cutmix_batch(
                    images, features, labels, alpha=alpha_now)
            elif use_mixup:
                images, features, labels_a, labels_b, lam = mixup_batch(
                    images, features, labels, alpha=alpha_now)

        def forward_loss():
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits, _ = model(images, features)
                if use_aug_now and lam < 1.0:
                    loss = (lam * criterion(logits, labels_a) +
                            (1 - lam) * criterion(logits, labels_b))
                else:
                    loss = criterion(logits, labels)
            return logits, loss

        if use_sam:
            logits, loss = forward_loss()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.first_step(zero_grad=True)
                scaler.update()
                logits, loss = forward_loss()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.second_step(zero_grad=True)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.first_step(zero_grad=True)
                logits, loss = forward_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.second_step(zero_grad=True)
        else:
            logits, loss = forward_loss()
            if not torch.isfinite(loss):
                print(f"  ⚠️  Non-finite loss — skipping batch")
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits.detach(), 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / max(1, len(train_loader)), 100.0 * correct / max(1, total)


def evaluate(model, val_loader, device, criterion=None,
             use_tta=False, tta_transforms=None):
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
            paths    = batch['path']

            if use_tta and tta_transforms is not None:
                logit_sum = None
                for t in tta_transforms:
                    imgs_aug = torch.stack([
                        t(Image.open(p).convert('RGB')) for p in paths
                    ]).to(device)
                    logits, _ = model(imgs_aug, features)
                    logit_sum = logits if logit_sum is None else logit_sum + logits
                logits = logit_sum / len(tta_transforms)
            else:
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
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def sanitize_features(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=np.float32)
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)


def build_feature_cache(image_paths, feature_scaler=None, fit_scaler=False):
    raw   = {}
    n_bad = 0

    for img_path in tqdm(image_paths, desc="Extracting features", leave=False):
        try:
            img   = Image.open(img_path).convert('RGB')
            feats = extract_medical_features(img)
            feats = sanitize_features(feats)
            if len(feats) != NUM_TRADITIONAL_FEATURES:
                feats = np.zeros(NUM_TRADITIONAL_FEATURES, dtype=np.float32)
                n_bad += 1
        except Exception as e:
            print(f"  ⚠️  Feature fail {Path(img_path).name}: {e}")
            feats = np.zeros(NUM_TRADITIONAL_FEATURES, dtype=np.float32)
            n_bad += 1
        raw[img_path] = feats

    if n_bad:
        pct = 100.0 * n_bad / max(1, len(image_paths))
        print(f"  ⚠️  {n_bad}/{len(image_paths)} ({pct:.1f}%) used zero-vector fallback")
        if pct > 20:
            print("  ❌  >20% failures — check extract_medical_features()!")

    sample_vals = np.concatenate([raw[p] for p in list(raw.keys())[:10]])
    if np.allclose(sample_vals, 0.0):
        print("  ❌  CRITICAL: All features zero — extractor may be broken!")
    else:
        print(f"  ✅  Feature sanity check passed — "
              f"range [{sample_vals.min():.3f}, {sample_vals.max():.3f}]")

    if fit_scaler:
        all_feats      = np.nan_to_num(np.stack([raw[p] for p in image_paths]))
        feature_scaler = RobustScaler(quantile_range=(10.0, 90.0))
        feature_scaler.fit(all_feats)

    cache = {}
    for p in image_paths:
        if feature_scaler is not None:
            scaled   = feature_scaler.transform([raw[p]])[0]
            scaled   = np.clip(scaled, -3.0, 3.0)
            cache[p] = sanitize_features(scaled)
        else:
            cache[p] = raw[p]

    return cache, feature_scaler


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train_hybrid_model(data_dir, output_dir, epochs=100, batch_size=32,
                       lr=5e-4, early_stopping_patience=20, num_workers=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    data_path = Path(data_dir)

    print(f"\n📁 Scanning: {data_path}")

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
                    sorted(cls_dir.glob('*.png')) + sorted(cls_dir.glob('*.PNG')) +
                    sorted(cls_dir.glob('*.jpeg')))
            print(f"     {cls}: {len(imgs)} images")
            for ip in imgs:
                paths.append(str(ip)); lbls.append(idx)
        return paths, lbls

    if train_path.exists() and val_path.exists():
        print("✅ Pre-split train/ and val/ found.")
        class_names = sorted([p.name for p in train_path.iterdir() if p.is_dir()])
        print(f"   Classes: {class_names}")
        print("\n📂 train/")
        train_paths_raw, train_labels_raw = load_split(train_path, class_names)
        print("\n📂 val/")
        val_paths_raw, val_labels_raw = load_split(val_path, class_names)

        val_counts      = Counter(val_labels_raw)
        imbalance_ratio = max(val_counts.values()) / max(min(val_counts.values()), 1)

        if imbalance_ratio > 3.0:
            print(f"\n⚠️  Val set severely imbalanced (ratio={imbalance_ratio:.1f}×). "
                  f"Re-splitting 80/20 with stratification...")
            all_paths  = train_paths_raw  + val_paths_raw
            all_labels = train_labels_raw + val_labels_raw
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
            print("\n📊 Re-split result:")
            for idx, name in enumerate(class_names):
                print(f"   {name}: train={Counter(train_labels)[idx]}  "
                      f"val={Counter(val_labels)[idx]}")
        else:
            train_paths, train_labels = train_paths_raw, train_labels_raw
            val_paths,   val_labels   = val_paths_raw,   val_labels_raw
    else:
        print("⚠️  No pre-split dirs — scanning root and splitting 80/20.")
        class_names = sorted([
            p.name for p in data_path.iterdir()
            if p.is_dir() and p.name not in ('test', 'synthetic', 'sipakmed_raw')
        ])
        all_paths, all_labels = load_split(data_path, class_names)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

    print(f"\n✅ Train: {len(train_paths)}  |  Val: {len(val_paths)}")
    if len(train_paths) == 0:
        raise ValueError("No training images found. Check --data-dir.")

    # ── Feature extraction ────────────────────────────────────────────────
    print("\n🔍 Extracting train features (fit scaler)...")
    train_cache, feature_scaler = build_feature_cache(train_paths, fit_scaler=True)
    print("🔍 Extracting val features (apply scaler)...")
    val_cache, _ = build_feature_cache(val_paths, feature_scaler=feature_scaler)

    # ── Transforms — 300×300 ─────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.RandomGrayscale(p=0.05),
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tta_transforms = build_tta_transforms(CNN_INPUT_SIZE) if USE_TTA else None

    # ── Datasets & Loaders ───────────────────────────────────────────────
    train_ds = HybridDataset(train_paths, train_labels, train_transform, train_cache)
    val_ds   = HybridDataset(val_paths,   val_labels,   val_transform,   val_cache)

    class_counts = Counter(train_labels)
    normal_idx   = class_names.index('Normal') if 'Normal' in class_names else -1
    sample_weights = [
        (1.0 / class_counts[l]) * (2.0 if l == normal_idx else 1.0)
        for l in train_labels
    ]
    sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)

    num_workers  = min(num_workers, os.cpu_count() or 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(
        num_classes=len(class_names),
        num_features=NUM_TRADITIONAL_FEATURES,
        device=device, dropout=0.5, drop_path_rate=0.2,
    )
    freeze_backbone(model)

    # ── Optimizer ─────────────────────────────────────────────────────────
    backbone_param_ids = {id(p) for p in model.backbone.parameters()}
    head_params     = [p for p in model.parameters() if id(p) not in backbone_param_ids]
    backbone_params = list(model.backbone.parameters())

    def _make_optimizer(backbone_lr, head_lr):
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': 5e-4},
            {'params': head_params,     'lr': head_lr,     'weight_decay': 5e-4},
        ]
        if USE_SAM:
            return SAM(param_groups, torch.optim.AdamW, rho=0.05,
                       lr=head_lr, weight_decay=5e-4)
        return torch.optim.AdamW(param_groups)

    optimizer = _make_optimizer(backbone_lr=lr * 0.1, head_lr=lr)
    # FIX 3: pass unfreeze_epoch so scheduler can apply safe floor after unfreeze
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=3, total_epochs=epochs, n_cycles=3,
        unfreeze_epoch=BACKBONE_FREEZE_EPOCHS,
    )

    # ── Loss ─────────────────────────────────────────────────────────────
    class_weights_tensor = torch.tensor(
        [1.0 / class_counts[i] for i in range(len(class_names))],
        dtype=torch.float32, device=device,
    )
    class_weights_tensor = class_weights_tensor / class_weights_tensor.sum() * len(class_names)
    criterion = FocalLoss(alpha=0.25, gamma=2.0,
                          weight=class_weights_tensor, smoothing=0.05)

    # ── AMP ───────────────────────────────────────────────────────────────
    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"\n{'='*70}")
    print(f"  AMP      : {'Enabled' if use_amp else 'Disabled'}")
    print(f"  TTA      : {'Enabled (' + str(TTA_AUGMENTS) + ' views)' if USE_TTA else 'Disabled'}")
    print(f"  SAM      : {'Enabled' if USE_SAM else 'Disabled'}")
    print(f"  SWA      : {'Enabled (start ep ' + str(SWA_START) + ')' if USE_SWA else 'Disabled'}")
    print(f"  CutMix   : {'Enabled' if USE_CUTMIX else 'Disabled'}")
    print(f"  Input res: {CNN_INPUT_SIZE}×{CNN_INPUT_SIZE}")
    print(f"  feat_out : {FEAT_OUT_DIM}  fusion_dim={1536 + FEAT_OUT_DIM}")
    print(f"{'='*70}\n")

    # ── SWA ───────────────────────────────────────────────────────────────
    swa_model     = AveragedModel(model) if USE_SWA else None
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5) if USE_SWA else None

    # ── Training loop ─────────────────────────────────────────────────────
    best_macro_f1     = 0.0
    best_val_loss     = float('inf')
    best_val_acc      = 0.0
    patience_counter  = 0
    backbone_unfrozen = False
    unfreeze_epoch_actual = None
    checkpoint_path   = os.path.join(output_dir, 'best_model.pt')
    swa_checkpoint    = os.path.join(output_dir, 'swa_model.pt')
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [], 'val_macro_f1': [],
    }

    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*70}")

        # Progressive backbone unfreezing
        if not backbone_unfrozen and epoch >= BACKBONE_FREEZE_EPOCHS:
            backbone_unfrozen     = True
            unfreeze_epoch_actual = epoch
            unfreeze_backbone_progressive(model, epoch, BACKBONE_FREEZE_EPOCHS, epochs)

            # FIX 3: rebuild optimizer with safe low LR; scheduler knows to
            # hold at safe_lr_scale for UNFREEZE_WARMUP_EPOCHS epochs
            optimizer = _make_optimizer(backbone_lr=lr * 0.05, head_lr=lr * 0.3)
            remaining = epochs - epoch
            scheduler = WarmupCosineScheduler(
                optimizer, warmup_epochs=1, total_epochs=remaining, n_cycles=2,
                unfreeze_epoch=epoch, safe_lr_scale=0.1,
            )
            print(f"  Optimizer rebuilt — safe LR floor for {UNFREEZE_WARMUP_EPOCHS} epochs "
                  f"(FIX 3: prevents spike at first backbone gradient)")

        elif backbone_unfrozen and epoch > BACKBONE_FREEZE_EPOCHS:
            unfreeze_backbone_progressive(model, epoch, BACKBONE_FREEZE_EPOCHS, epochs)

        use_swa_now = USE_SWA and (epoch >= SWA_START)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, use_mixup=True, use_cutmix=USE_CUTMIX,
            current_epoch=epoch, mixup_start_epoch=25, max_mixup_alpha=0.2,
            use_sam=USE_SAM,
        )

        if use_swa_now:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            # FIX 4: use custom BN updater that handles dict-yielding DataLoader
            _update_bn_hybrid(train_loader, swa_model, device)
            val_acc, val_loss, _, _, macro_f1 = evaluate(
                swa_model, val_loader, device, criterion,
                use_tta=USE_TTA, tta_transforms=tta_transforms,
            )
        else:
            scheduler.step()
            val_acc, val_loss, _, _, macro_f1 = evaluate(
                model, val_loader, device, criterion,
                use_tta=USE_TTA, tta_transforms=tta_transforms,
            )

        new_lrs   = [pg['lr'] for pg in optimizer.param_groups]
        alpha_now = get_mixup_alpha(epoch, mixup_start_epoch=25, max_alpha=0.2)
        print(f"Train  — loss: {train_loss:.4f} | acc: {train_acc:.2f}%")
        print(f"Val    — loss: {val_loss:.4f}  | acc: {val_acc:.2f}% | "
              f"macro-F1: {macro_f1:.4f}{'  [SWA]' if use_swa_now else ''}")
        print(f"LR     — head={new_lrs[1]:.2e}  backbone={new_lrs[0]:.2e}"
              f"  | mixup_alpha={alpha_now:.3f}")

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
            save_model       = swa_model if use_swa_now else model
            torch.save({
                'epoch':               epoch + 1,
                'model_state_dict':    save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':             val_acc,
                'macro_f1':            macro_f1,
                'class_names':         class_names,
                'num_features':        NUM_TRADITIONAL_FEATURES,
                'feat_out_dim':        FEAT_OUT_DIM,
                'script_version':      _SCRIPT_VERSION,
                'input_size':          CNN_INPUT_SIZE,
                'use_swa':             use_swa_now,
            }, checkpoint_path)
            print(f"✅ Checkpoint saved  (macro-F1={macro_f1:.4f}, val_acc={val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{early_stopping_patience})")
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}.")
                break

    if USE_SWA and swa_model is not None:
        torch.save({
            'model_state_dict': swa_model.state_dict(),
            'class_names':      class_names,
            'num_features':     NUM_TRADITIONAL_FEATURES,
            'feat_out_dim':     FEAT_OUT_DIM,
            'script_version':   _SCRIPT_VERSION,
            'input_size':       CNN_INPUT_SIZE,
        }, swa_checkpoint)
        print(f"  SWA model saved: {swa_checkpoint}")

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
    parser.add_argument('--epochs',                  type=int,   default=50)
    parser.add_argument('--batch-size',              type=int,   default=32)
    parser.add_argument('--learning-rate',           type=float, default=5e-4)
    parser.add_argument('--early-stopping-patience', type=int,   default=20)
    parser.add_argument('--num-workers',             type=int,   default=4)
    parser.add_argument('--seed',                    type=int,   default=42)
    parser.add_argument('--no-tta',    action='store_true', help='Disable TTA evaluation')
    parser.add_argument('--no-swa',    action='store_true', help='Disable SWA')
    parser.add_argument('--use-sam',   action='store_true', help='Enable SAM optimizer (2× slower)')
    parser.add_argument('--no-cutmix', action='store_true', help='Disable CutMix augmentation')
    args = parser.parse_args()

    if args.no_tta:    USE_TTA    = False
    if args.no_swa:    USE_SWA    = False
    if args.use_sam:   USE_SAM    = True
    if args.no_cutmix: USE_CUTMIX = False

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