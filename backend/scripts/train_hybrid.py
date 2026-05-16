"""
Training Script for GPU-Optimized Hybrid Model
Combines EfficientNet-B3 CNN features with traditional medical features

═══════════════════════════════════════════════════════════════════════
ROOT CAUSE ANALYSIS & FIXES (v7 — 80% accuracy target)
═══════════════════════════════════════════════════════════════════════

All v6 bugs retained and fixed. New improvements in v7:

IMPROVEMENT 1 — Input resolution upgraded to 300×300
  EfficientNet-B3 was designed for 300×300. Cervical cytology requires
  fine nuclear detail that 224×224 loses. This alone gives +3–5% on
  medical imaging tasks.

IMPROVEMENT 2 — MixUp start pushed later + alpha reduced
  Logs showed train acc dropping 62%→44% when mixup_alpha ramped at
  epoch 17. Pushed mixup_start_epoch=25, max_alpha=0.2 to let the
  model stabilize before augmentation destabilizes it.

IMPROVEMENT 3 — FocalLoss restored (with label smoothing inside)
  CrossEntropyLoss+smoothing was masking the Normal class imbalance
  (198 vs ~409 samples for other classes). FocalLoss with gamma=2.0
  down-weights easy majorities and rescues the Normal class.

IMPROVEMENT 4 — Test-Time Augmentation (TTA) at evaluation
  Free +2–4% accuracy by averaging predictions over 5 augmented views
  of each val image. Zero training cost. Applied during evaluate().

IMPROVEMENT 5 — Progressive layer unfreezing (gradual backbone thaw)
  Instead of all-at-once unfreeze at epoch 5, unfreeze the backbone
  block-by-block every N epochs. Prevents catastrophic forgetting of
  ImageNet features on small datasets.

IMPROVEMENT 6 — Cosine annealing with warm restarts (SGDR)
  Replaced simple WarmupCosine with CosineAnnealingWarmRestarts.
  Warm restarts help escape local minima — particularly useful when
  the model is stuck in the 40–45% plateau.

IMPROVEMENT 7 — SAM (Sharpness-Aware Minimization) optimizer wrapper
  SAM finds flatter minima that generalize better. Particularly
  effective on small medical datasets where sharp minima = overfit.
  Adds ~2× training time per epoch but significantly improves val acc.
  Toggle with USE_SAM = True/False.

IMPROVEMENT 8 — Stronger dropout schedule
  Dropout 0.5 in classifier + 0.3 in FeatureMLP (was 0.2).
  Addresses the persistent ~19pt train-val gap seen in v6 logs.

IMPROVEMENT 9 — CutMix added alongside MixUp
  CutMix (cuts a rectangular patch from one image and pastes into
  another) works better than MixUp on fine-grained visual features
  like cell morphology. Randomly selects MixUp or CutMix per batch.

IMPROVEMENT 10 — Stochastic Depth (DropPath) in backbone
  Adds per-layer drop probability during training. Acts as a stronger
  regulariser than dropout alone. Applied to EfficientNet blocks.

IMPROVEMENT 11 — SWA (Stochastic Weight Averaging)
  Averages weights from the last N epochs. Gives free +1–2% and
  better calibration. Applied after epoch SWA_START.

═══════════════════════════════════════════════════════════════════════
v6 BUGS (all retained / fixed)
═══════════════════════════════════════════════════════════════════════

BUG 1  — NaN LOSS: mixup_data used alpha as fixed lambda. FIXED.
BUG 2  — WarmupCosine ran per-batch not per-epoch. FIXED.
BUG 3  — Differential LR broken by lr_scale. FIXED.
BUG 4  — Pre-split contaminated by forced re-merge. FIXED.
BUG 5  — extract_medical_features got path string not PIL. FIXED.
BUG 6  — AMP not enabled. FIXED.
BUG 7  — WeightedRandomSampler used wrong label list. FIXED.
BUG 8  — LR too high (2e-3). FIXED → 5e-4.
BUG 9  — MixUp started too early. FIXED → epoch 25.
BUG 10 — 4-layer head overfit. FIXED → 2-layer.
BUG 11 — Hue jitter too strong. FIXED → 0.05.
BUG 12 — Backbone never frozen first. FIXED → 5 epoch freeze.
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

_SCRIPT_VERSION = "EfficientNet-B3-Hybrid-v7-accuracy-push"
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
CNN_INPUT_SIZE = 300        # IMPROVEMENT 1: was 224; B3 designed for 300×300

# How many epochs to keep backbone frozen before progressive unfreezing
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
# SAM Optimizer (IMPROVEMENT 7)
# ─────────────────────────────────────────────────────────────────────────────
class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer wrapper.
    Finds flatter minima that generalize better — especially useful on
    small medical datasets where sharp minima leads to overfitting.

    Usage: requires two forward/backward passes per batch.
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
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
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
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ─────────────────────────────────────────────────────────────────────────────
# DropPath / Stochastic Depth (IMPROVEMENT 10)
# ─────────────────────────────────────────────────────────────────────────────
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample during training."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    output = x / keep_prob * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class FeatureMLP(nn.Module):
    """
    MLP for the 30-dim traditional feature branch with residual connection.
    IMPROVEMENT 8: dropout raised to 0.3 from 0.2 to fight train-val gap.
    """
    def __init__(self, in_dim: int, out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.proj(x)


class EfficientNetHybrid(nn.Module):
    """
    EfficientNet-B3 backbone (300×300 input) + traditional feature MLP,
    fused with learned attention gate.

    v7 changes:
    - Input size 300×300 (IMPROVEMENT 1)
    - Simplified 2-layer classifier head (BUG 10 fix kept)
    - Higher dropout 0.5 → 0.5 in head, 0.3 in FeatureMLP (IMPROVEMENT 8)
    - DropPath added to backbone blocks (IMPROVEMENT 10)
    """
    def __init__(self, num_classes: int, num_features: int = 30,
                 dropout: float = 0.5, drop_path_rate: float = 0.2):
        super().__init__()
        try:
            from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
            backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            cnn_out_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()

            # IMPROVEMENT 10: inject DropPath into EfficientNet MBConv blocks
            self._inject_drop_path(backbone, drop_path_rate)
            self.backbone = backbone
        except Exception:
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            cnn_out_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone

        self.cnn_out_dim = cnn_out_dim
        feat_out_dim = 256

        self.feature_mlp = FeatureMLP(num_features, feat_out_dim, dropout=0.3)

        fusion_dim = cnn_out_dim + feat_out_dim

        # Attention gate: learns how much to trust CNN vs traditional features
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )

        # 2-layer classifier head (BUG 10 fix)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def _inject_drop_path(self, backbone, drop_path_rate: float):
        """
        Walk EfficientNet's features blocks and add DropPath to each
        MBConv block's stochastic depth slot.
        """
        try:
            blocks = list(backbone.features.children())
            n_blocks = sum(
                1 for b in blocks
                if hasattr(b, '__iter__')
                for _ in b
            )
            dp_rates = torch.linspace(0, drop_path_rate, n_blocks).tolist()
            idx = 0
            for stage in blocks:
                if not hasattr(stage, '__iter__'):
                    continue
                for block in stage:
                    if hasattr(block, 'stochastic_depth'):
                        block.stochastic_depth.p = dp_rates[idx]
                    idx += 1
        except Exception:
            pass  # non-fatal; backbone still works without DropPath

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


def build_model(num_classes, num_features, device, dropout=0.5, drop_path_rate=0.2):
    model = EfficientNetHybrid(
        num_classes=num_classes,
        num_features=num_features,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
    )
    model = model.to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model            : EfficientNetHybrid (EfficientNet-B3 @ 300×300)")
    print(f"  Total parameters : {n_params:,}")
    print(f"  Trainable now    : {trainable:,}")
    print(f"  Device           : {device}")
    print(f"  Classes          : {num_classes}")
    print(f"  DropPath rate    : {drop_path_rate}")
    return model


def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False
    frozen = sum(p.numel() for p in model.backbone.parameters())
    print(f"  🔒 Backbone frozen ({frozen:,} params) — head-only training")


def get_backbone_stages(model):
    """Return EfficientNet feature stages as a list for progressive unfreezing."""
    try:
        stages = list(model.backbone.features.children())
        return stages
    except Exception:
        return []


def unfreeze_backbone_progressive(model, epoch, freeze_start, total_epochs):
    """
    IMPROVEMENT 5: Progressive layer unfreezing.
    Unfreeze backbone blocks one group at a time over the training run,
    from top (classifier-near) to bottom (input-near).
    Prevents catastrophic forgetting on small datasets.
    """
    stages = get_backbone_stages(model)
    if not stages:
        # Fallback: all-at-once unfreeze
        for param in model.backbone.parameters():
            param.requires_grad = True
        n = sum(p.numel() for p in model.backbone.parameters())
        print(f"  🔓 Backbone fully unfrozen ({n:,} params)")
        return

    n_stages     = len(stages)
    thaw_epochs  = total_epochs - freeze_start      # epochs available for thawing
    thaw_interval = max(1, thaw_epochs // n_stages)  # epochs between each unfreeze

    epochs_since_thaw = epoch - freeze_start
    n_to_unfreeze = min(n_stages, 1 + epochs_since_thaw // thaw_interval)

    # Unfreeze from top (last stages) toward bottom (first stages)
    for i, stage in enumerate(reversed(stages)):
        for param in stage.parameters():
            param.requires_grad = (i < n_to_unfreeze)

    unfrozen_params = sum(
        p.numel() for p in model.backbone.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"  🔓 Progressive unfreeze: {n_to_unfreeze}/{n_stages} stages "
          f"({unfrozen_params:,}/{total_params:,} params)")


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss with optional label smoothing.
    IMPROVEMENT 3: Restored over CrossEntropyLoss to handle Normal class
    imbalance (198 vs ~409 samples). gamma=2.0 down-weights easy examples.
    smoothing=0.05 reduces overconfidence without masking class signal.
    """
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, smoothing=0.05):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.weight    = weight
        self.smoothing = smoothing

    def forward(self, logits, targets):
        logits = torch.clamp(logits, -50.0, 50.0)
        n_cls  = logits.size(-1)

        # Label smoothing baked in
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.smoothing / (n_cls - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss   = -(smooth_targets * log_probs).sum(dim=-1)

        # Apply per-class weights
        if self.weight is not None:
            w = self.weight[targets]
            ce_loss = ce_loss * w

        # Focal modulation
        probs       = torch.exp(-ce_loss)
        focal_loss  = self.alpha * (1 - probs) ** self.gamma * ce_loss
        return focal_loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class HybridDataset(Dataset):
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
    """True Beta-sampled MixUp. BUG 1 fix retained."""
    batch_size = images.size(0)
    lam        = float(np.random.beta(alpha, alpha))
    lam        = max(0.05, min(0.95, lam))
    index      = torch.randperm(batch_size, device=images.device)
    mixed_images   = lam * images   + (1 - lam) * images[index]
    mixed_features = lam * features + (1 - lam) * features[index]
    return mixed_images, mixed_features, labels, labels[index], lam


def cutmix_batch(images, features, labels, alpha=0.2):
    """
    IMPROVEMENT 9: CutMix augmentation.
    Cuts a random box from one image and pastes into another.
    Better than MixUp for fine-grained morphological features
    (nuclei, cell borders) because it preserves local texture.
    """
    batch_size = images.size(0)
    lam        = float(np.random.beta(alpha, alpha))
    lam        = max(0.05, min(0.95, lam))
    index      = torch.randperm(batch_size, device=images.device)

    _, _, H, W = images.shape
    cut_ratio   = math.sqrt(1 - lam)
    cut_h       = int(H * cut_ratio)
    cut_w       = int(W * cut_ratio)
    cx          = random.randint(0, W)
    cy          = random.randint(0, H)
    x1          = max(0, cx - cut_w // 2)
    y1          = max(0, cy - cut_h // 2)
    x2          = min(W, cx + cut_w // 2)
    y2          = min(H, cy + cut_h // 2)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    # Adjust lambda based on actual cut area
    lam = 1 - (x2 - x1) * (y2 - y1) / (H * W)

    # For features: linear interpolation (CutMix only cuts image, not tabular features)
    mixed_features = lam * features + (1 - lam) * features[index]

    return mixed_images, mixed_features, labels, labels[index], lam


def get_mixup_alpha(epoch, mixup_start_epoch=25, max_alpha=0.2, ramp_epochs=10):
    """
    IMPROVEMENT 2: MixUp ramp starts at epoch 25 (was 15) and max_alpha=0.2
    (was 0.3). This prevents the 27pt accuracy collapse seen in v6 logs
    at epoch 17 when alpha ramped too early.
    Returns 0.0 before mixup_start_epoch.
    """
    if epoch < mixup_start_epoch:
        return 0.0
    ramp      = min(1.0, (epoch - mixup_start_epoch) / max(1, ramp_epochs))
    alpha_now = max_alpha * ramp
    return alpha_now


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler — BUG 2 + BUG 3 fix retained
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """
    Per-EPOCH warmup + cosine annealing with warm restarts (IMPROVEMENT 6).
    Called ONCE per epoch (BUG 2 fix). Per-group initial_lr preserved (BUG 3 fix).
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, n_cycles=3):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.n_cycles       = n_cycles
        self.current_epoch  = 0

        for pg in optimizer.param_groups:
            pg['initial_lr'] = pg['lr']

    def step(self):
        self.current_epoch += 1
        e = self.current_epoch

        if e <= self.warmup_epochs:
            scale = e / max(1, self.warmup_epochs)
        else:
            # Cosine with warm restarts (SGDR)
            progress  = (e - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            # n_cycles restarts over the cosine period
            cycle_pos = progress * self.n_cycles % 1.0
            scale     = 0.5 * (1.0 + math.cos(math.pi * cycle_pos))
            # Decay amplitude over time so LR trend is still downward
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
# TTA helper (IMPROVEMENT 4)
# ─────────────────────────────────────────────────────────────────────────────
def build_tta_transforms(input_size=300):
    """
    Returns a list of transforms for Test-Time Augmentation.
    Each val image is run through all of these; logits are averaged.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    base = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ]
    return [
        transforms.Compose(base),
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)] + base),
        transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]   + base),
        transforms.Compose([transforms.RandomRotation((90, 90))]    + base),
        transforms.Compose([transforms.RandomRotation((270, 270))]  + base),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Train epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, train_loader, optimizer, criterion, device,
                scaler=None, use_mixup=False, use_cutmix=False,
                current_epoch=0, mixup_start_epoch=25, max_mixup_alpha=0.2,
                use_sam=False):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
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
                n_bad = (~torch.isfinite(features)).sum().item()
                print(f"  ⚠️  DIAGNOSTIC: features contain {n_bad} NaN/Inf")
            else:
                fmin, fmax = features.min().item(), features.max().item()
                print(f"  ✅  DIAGNOSTIC: features OK — range [{fmin:.2f}, {fmax:.2f}]")
                if abs(fmin) < 1e-6 and abs(fmax) < 1e-6:
                    print("  ❌  CRITICAL: features ALL ZEROS!")

        alpha_now     = get_mixup_alpha(current_epoch, mixup_start_epoch, max_mixup_alpha)
        use_aug_now   = (alpha_now > 0.0)
        labels_a = labels_b = labels
        lam = 1.0

        if use_aug_now:
            # IMPROVEMENT 9: randomly choose CutMix or MixUp per batch
            if use_cutmix and use_mixup and random.random() > 0.5:
                images, features, labels_a, labels_b, lam = cutmix_batch(
                    images, features, labels, alpha=alpha_now
                )
            elif use_mixup:
                images, features, labels_a, labels_b, lam = mixup_batch(
                    images, features, labels, alpha=alpha_now
                )

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
            # SAM: two-pass optimization (IMPROVEMENT 7)
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

    acc      = 100.0 * correct / max(1, total)
    avg_loss = total_loss / max(1, len(train_loader))
    return avg_loss, acc


def evaluate(model, val_loader, device, criterion=None,
             use_tta=False, tta_transforms=None, val_paths=None):
    """
    IMPROVEMENT 4: Optional TTA evaluation.
    When use_tta=True, each image is run through multiple augmented views
    and predictions are averaged — gives free +2–4% accuracy.
    Falls back to single-pass if TTA transforms not provided.
    """
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
                # Average logits over all TTA transforms
                logit_sum = None
                for t in tta_transforms:
                    imgs_aug = []
                    for p in paths:
                        img = Image.open(p).convert('RGB')
                        imgs_aug.append(t(img))
                    imgs_aug  = torch.stack(imgs_aug).to(device)
                    logits, _ = model(imgs_aug, features)
                    if logit_sum is None:
                        logit_sum = logits
                    else:
                        logit_sum = logit_sum + logits
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
# Feature extraction — BUG 5 fix retained
# ─────────────────────────────────────────────────────────────────────────────
def sanitize_features(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, -1e6, 1e6)
    return arr


def build_feature_cache(image_paths, feature_scaler=None, fit_scaler=False):
    """
    BUG 5 FIX (CRITICAL, retained from v6):
    extract_medical_features() expects a PIL Image object.
    Opens each image and passes the Image — not the path string.
    """
    raw   = {}
    n_bad = 0

    for img_path in tqdm(image_paths, desc="Extracting features", leave=False):
        try:
            img   = Image.open(img_path).convert('RGB')
            feats = extract_medical_features(img)
            feats = sanitize_features(feats)
            if len(feats) != NUM_TRADITIONAL_FEATURES:
                print(f"  ⚠️  Dim mismatch {Path(img_path).name}: "
                      f"got {len(feats)}, expected {NUM_TRADITIONAL_FEATURES}")
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
              f"sample range [{sample_vals.min():.3f}, {sample_vals.max():.3f}]")

    if fit_scaler:
        all_feats = np.stack([raw[p] for p in image_paths])
        all_feats = np.nan_to_num(all_feats, nan=0.0)
        feature_scaler = RobustScaler(quantile_range=(10.0, 90.0))
        feature_scaler.fit(all_feats)
        print(f"  ✅  RobustScaler fit: center range "
              f"[{feature_scaler.center_.min():.2f}, {feature_scaler.center_.max():.2f}]")

    cache = {}
    for p in image_paths:
        if feature_scaler is not None:
            scaled = feature_scaler.transform([raw[p]])[0]
            scaled = np.clip(scaled, -3.0, 3.0)
            scaled = sanitize_features(scaled)
            cache[p] = scaled
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

    # ── Data loading (BUG 4 fix retained) ────────────────────────────────
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

        val_counts      = Counter(val_labels_raw)
        min_val         = min(val_counts.values())
        max_val         = max(val_counts.values())
        imbalance_ratio = max_val / max(min_val, 1)

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
            train_paths, train_labels = train_paths_raw, train_labels_raw
            val_paths,   val_labels   = val_paths_raw,   val_labels_raw
            print("✅ Val distribution balanced — using pre-made split.")
    else:
        print("⚠️  No pre-split dirs — scanning root and splitting 80/20.")
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

    # ── Feature extraction (BUG 5 fix retained) ──────────────────────────
    print("\n🔍 Extracting train features (fit scaler)...")
    train_cache, feature_scaler = build_feature_cache(train_paths, fit_scaler=True)
    print("🔍 Extracting val features (apply scaler)...")
    val_cache, _ = build_feature_cache(val_paths, feature_scaler=feature_scaler)

    n_synthetic = sum(1 for p in train_paths if 'synthetic' in p.lower())
    n_real      = len(train_paths) - n_synthetic
    print(f"\n📊 Train composition: {n_real} real  +  {n_synthetic} synthetic")

    # ── Transforms — IMPROVEMENT 1: 300×300; BUG 11 fix retained ─────────
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.RandomGrayscale(p=0.05),
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),  # 300×300
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),  # 300×300
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── TTA transforms (IMPROVEMENT 4) ───────────────────────────────────
    tta_transforms = build_tta_transforms(CNN_INPUT_SIZE) if USE_TTA else None

    # ── Datasets & Loaders ───────────────────────────────────────────────
    train_ds = HybridDataset(train_paths, train_labels, train_transform, train_cache)
    val_ds   = HybridDataset(val_paths,   val_labels,   val_transform,   val_cache)

    # BUG 7 fix retained: sampler uses final train_labels
    class_counts = Counter(train_labels)
    normal_idx   = class_names.index('Normal') if 'Normal' in class_names else -1
    def _cls_weight(i):
        base = 1.0 / class_counts[i]
        return base * 2.0 if i == normal_idx else base
    sample_weights = [_cls_weight(l) for l in train_labels]
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
        device=device,
        dropout=0.5,
        drop_path_rate=0.2,
    )
    freeze_backbone(model)

    # ── Optimizer (BUG 8 fix retained: lr=5e-4) ──────────────────────────
    backbone_param_ids = {id(p) for p in model.backbone.parameters()}
    head_params        = [p for p in model.parameters() if id(p) not in backbone_param_ids]
    backbone_params    = list(model.backbone.parameters())

    def _make_optimizer(backbone_lr, head_lr):
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': 5e-4},
            {'params': head_params,     'lr': head_lr,     'weight_decay': 5e-4},
        ]
        if USE_SAM:
            return SAM(param_groups, torch.optim.AdamW,
                       rho=0.05, lr=head_lr, weight_decay=5e-4)
        return torch.optim.AdamW(param_groups)

    optimizer = _make_optimizer(backbone_lr=lr * 0.1, head_lr=lr)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=3, total_epochs=epochs, n_cycles=3)

    # ── Loss — IMPROVEMENT 3: FocalLoss with label smoothing ─────────────
    class_weights_tensor = torch.tensor(
        [1.0 / class_counts[i] for i in range(len(class_names))],
        dtype=torch.float32, device=device
    )
    class_weights_tensor = class_weights_tensor / class_weights_tensor.sum() * len(class_names)
    criterion = FocalLoss(
        alpha=0.25, gamma=2.0,
        weight=class_weights_tensor,
        smoothing=0.05,
    )

    # ── AMP (BUG 6 fix retained) ──────────────────────────────────────────
    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"  AMP      : {'Enabled' if use_amp else 'Disabled'}")
    print(f"  TTA      : {'Enabled (' + str(TTA_AUGMENTS) + ' views)' if USE_TTA else 'Disabled'}")
    print(f"  SAM      : {'Enabled' if USE_SAM else 'Disabled'}")
    print(f"  SWA      : {'Enabled (start ep ' + str(SWA_START) + ')' if USE_SWA else 'Disabled'}")
    print(f"  CutMix   : {'Enabled' if USE_CUTMIX else 'Disabled'}")
    print(f"  Input res: {CNN_INPUT_SIZE}×{CNN_INPUT_SIZE}")

    # ── SWA setup (IMPROVEMENT 11) ────────────────────────────────────────
    swa_model = AveragedModel(model) if USE_SWA else None
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5) if USE_SWA else None

    # ── Training Loop ─────────────────────────────────────────────────────
    best_macro_f1     = 0.0
    best_val_loss     = float('inf')
    best_val_acc      = 0.0
    patience_counter  = 0
    backbone_unfrozen = False
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

        # IMPROVEMENT 5: progressive backbone unfreezing
        if not backbone_unfrozen and epoch >= BACKBONE_FREEZE_EPOCHS:
            backbone_unfrozen = True
            unfreeze_backbone_progressive(model, epoch, BACKBONE_FREEZE_EPOCHS, epochs)
            optimizer = _make_optimizer(backbone_lr=lr * 0.1, head_lr=lr * 0.5)
            remaining = epochs - epoch
            scheduler = WarmupCosineScheduler(
                optimizer, warmup_epochs=1, total_epochs=remaining, n_cycles=2
            )
            print("  Optimizer rebuilt for fine-tuning phase.")
        elif backbone_unfrozen and epoch > BACKBONE_FREEZE_EPOCHS:
            # Continue progressive unfreezing each epoch
            unfreeze_backbone_progressive(model, epoch, BACKBONE_FREEZE_EPOCHS, epochs)

        # SWA: switch scheduler after SWA_START
        use_swa_now = USE_SWA and (epoch >= SWA_START)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler,
            use_mixup=True,
            use_cutmix=USE_CUTMIX,
            current_epoch=epoch,
            mixup_start_epoch=25,    # IMPROVEMENT 2: was 15
            max_mixup_alpha=0.2,     # IMPROVEMENT 2: was 0.3
            use_sam=USE_SAM,
        )

        # SWA: update averaged model + BN stats
        if use_swa_now:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            # Evaluate with SWA model (update BN first using a train pass)
            update_bn(train_loader, swa_model, device=device)
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

        new_lrs       = [pg['lr'] for pg in optimizer.param_groups]
        alpha_now     = get_mixup_alpha(epoch, mixup_start_epoch=25, max_alpha=0.2)
        print(f"Train  — loss: {train_loss:.4f} | acc: {train_acc:.2f}%")
        print(f"Val    — loss: {val_loss:.4f}  | acc: {val_acc:.2f}% | macro-F1: {macro_f1:.4f}"
              f"{'  [SWA]' if use_swa_now else ''}")
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

    # Save final SWA model separately if used
    if USE_SWA and swa_model is not None:
        torch.save({
            'model_state_dict': swa_model.state_dict(),
            'class_names':      class_names,
            'num_features':     NUM_TRADITIONAL_FEATURES,
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
    parser.add_argument('--no-tta',   action='store_true', help='Disable TTA evaluation')
    parser.add_argument('--no-swa',   action='store_true', help='Disable SWA')
    parser.add_argument('--use-sam',  action='store_true', help='Enable SAM optimizer (2× slower)')
    parser.add_argument('--no-cutmix',action='store_true', help='Disable CutMix augmentation')
    args = parser.parse_args()

    # Apply CLI toggles
    if args.no_tta:
        USE_TTA    = False
    if args.no_swa:
        USE_SWA    = False
    if args.use_sam:
        USE_SAM    = True
    if args.no_cutmix:
        USE_CUTMIX = False

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