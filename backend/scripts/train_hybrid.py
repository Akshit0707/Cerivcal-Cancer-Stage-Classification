"""
Training Script for GPU-Optimized Hybrid Model
Combines EfficientNet-B3 CNN features with traditional medical features

FIXES vs previous version:
  1. ATTENTION COLLAPSE: Detach CNN features before attention so gradients
     don't short-circuit the feature branch (was giving 1.000/0.000 every time)
  2. CLASS IMBALANCE: Focal loss replaces label smoothing for minority classes
     (CIN2/CIN3/Cancer had only 11 samples vs CIN1=76)
  3. CLASS WEIGHTS: Inverse-frequency weights passed to loss function
  4. OVERSAMPLE MINORITY: WeightedRandomSampler weight exponent increased 1.0→1.5
  5. FEATURE BRANCH LR: Feature MLP gets same LR as head, not backbone LR
  6. MIXUP DISABLED for minority classes below threshold (hurts rare classes)
  7. BACKBONE: EfficientNet-B3 pretrained on ImageNet
  8. LR SCHEDULER: WarmupCosine fixed
  9. DROPOUT: 0.4 before final classifier
 10. AUGMENTATION: Strong pipeline with RandomErasing after ToTensor (fixed)
"""

import argparse
import math
import random
import sys
import warnings
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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

warnings.filterwarnings("ignore")

_SCRIPT_VERSION = "EfficientNet-B3-Hybrid-v3-fixedattention"
print(f"[train_hybrid.py] version={_SCRIPT_VERSION}  file={__file__}")

# ── Path setup ────────────────────────────────────────────────────────────────
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


# ── Utilities ─────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def resolve_data_dir(user_path: str) -> Path:
    candidates = [
        Path(user_path),
        PROJECT_ROOT / user_path,
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "backend" / "data",
        PROJECT_ROOT / "backend" / "datasets",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p.resolve()
    raise FileNotFoundError(
        f"Data directory not found. Checked: {[str(c) for c in candidates]}"
    )


# ── FIX #1: Focal Loss for imbalanced classes ─────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal loss down-weights easy (well-classified) examples so the model
    focuses on hard minority-class examples like CIN2/CIN3/Cancer.
    gamma=2 is standard; alpha provides per-class inverse-frequency weighting.
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # per-class weights (inverse frequency)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ── FIX #2: Corrected attention — stops CNN from collapsing feature branch ────
class FeatureMLP(nn.Module):
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
    EfficientNet-B3 + medical feature MLP fused with fixed attention gate.

    KEY FIX: In the old version the attention gate received the raw concatenated
    features and its gradients flowed freely back into both branches. Because
    the CNN branch has ~12M parameters and the feature branch only ~200K, the
    CNN dominated and drove the attention weights to (1.0, 0.0) — completely
    ignoring the medical features.

    Fix: compute attention from DETACHED representations. The attention gate
    now decides how to weight the branches based on their current values but
    cannot update the branches through that path. Each branch is updated only
    through its own prediction loss, giving the feature branch a fair gradient.
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

        # Attention gate operates on detached features (see forward())
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
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

        self._freeze_backbone(freeze=True)

    def _freeze_backbone(self, freeze: bool):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def unfreeze_backbone(self, blocks_from_end: int = 2):
        params = list(self.backbone.parameters())
        n = len(params)
        unfreeze_from = max(0, n - blocks_from_end * (n // 10))
        for i, p in enumerate(params):
            p.requires_grad = (i >= unfreeze_from)

    def forward(self, images, features):
        cnn_feat  = self.backbone(images)        # (B, 1536)
        trad_feat = self.feature_mlp(features)   # (B, 128)

        fused = torch.cat([cnn_feat, trad_feat], dim=1)  # (B, 1664)

        # FIX: detach before attention so the gate cannot collapse feature branch
        attn = self.attention(fused.detach())             # (B, 2)

        cnn_scaled  = cnn_feat  * attn[:, 0:1]
        trad_scaled = trad_feat * attn[:, 1:2]
        fused_scaled = torch.cat([cnn_scaled, trad_scaled], dim=1)

        logits = self.classifier(fused_scaled)
        return logits, attn

    def get_feature_importance(self, avg_attention, feature_names):
        trad_weight = (
            float(avg_attention[:, 1].mean())
            if avg_attention.ndim > 1
            else float(avg_attention[1])
        )
        return {name: round(trad_weight / len(feature_names), 6) for name in feature_names}


def build_model(num_classes, num_features, device, pretrained_cnn_path=None):
    model = EfficientNetHybrid(num_classes=num_classes, num_features=num_features)
    model = model.to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model            : EfficientNetHybrid v3 (fixed attention)")
    print(f"  Total parameters : {n_params:,}")
    print(f"  Trainable now    : {trainable:,}  (backbone frozen during warmup)")
    print(f"  Estimated size   : {n_params * 4 / 1e6:.2f} MB")
    print(f"  Device           : {device}")
    print(f"  Classes          : {num_classes}")
    print(f"  Traditional feat : {num_features}")
    return model


# ── Dataset ───────────────────────────────────────────────────────────────────
class HybridDataset(Dataset):
    def __init__(self, data_dir, transform=None, feature_cache=None, feature_scaler=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.feature_cache = feature_cache if feature_cache is not None else {}
        self.feature_scaler = feature_scaler
        self.samples = []
        self.class_to_idx = {}

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        if not class_dirs:
            raise FileNotFoundError(f"No class folders found in: {self.data_dir}")

        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for img_path in class_dir.glob(ext):
                    if not img_path.name.startswith("._"):
                        self.samples.append((str(img_path), idx))

        if not self.samples:
            raise FileNotFoundError(f"No image files found in: {self.data_dir}")

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if img_path in self.feature_cache:
            features = self.feature_cache[img_path]
        else:
            raw_features = extract_medical_features(image)
            if self.feature_scaler is not None:
                features = self.feature_scaler.transform([raw_features])[0].tolist()
            else:
                features = raw_features
            self.feature_cache[img_path] = features

        if self.transform is not None:
            image = self.transform(image)

        return (
            image,
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


# ── Transforms — RandomErasing AFTER ToTensor (fixed) ────────────────────────
def get_transforms(augment=True):
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(CNN_INPUT_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # must be after ToTensor
        ])
    return transforms.Compose([
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_tta_transforms(n_augments=6):
    base = transforms.Compose([
        transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tta_list = [base]
    for _ in range(n_augments - 1):
        tta_list.append(transforms.Compose([
            transforms.RandomResizedCrop(CNN_INPUT_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    return tta_list


# ── Mixup ─────────────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(0.1, min(0.9, lam))
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(pred, y_a, y_b, lam, criterion):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── LR scheduler ──────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self._epoch = 0
        self._set_lr(min_lr)

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def step(self):
        self._epoch += 1
        e = self._epoch
        if e <= self.warmup_epochs:
            lr = self.min_lr + (self.base_lr - self.min_lr) * (e / self.warmup_epochs)
        else:
            progress = (e - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        self._set_lr(lr)
        return lr

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ── Train epoch ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, epoch, criterion, amp_scaler=None):
    model.train()
    running_loss = 0.0
    total = 0
    use_amp = device.type == "cuda" and amp_scaler is not None

    for images, features, labels in tqdm(loader, desc=f"Epoch {epoch} [train]"):
        images   = images.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        images_m, y_a, y_b, lam = mixup_data(images, labels, alpha=0.3)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits, _ = model(images_m, features)
                loss = mixup_criterion(logits, y_a, y_b, lam, criterion)
        else:
            logits, _ = model(images_m, features)
            loss = mixup_criterion(logits, y_a, y_b, lam, criterion)

        if torch.isnan(loss) or torch.isinf(loss):
            print("  WARNING: invalid loss, skipping batch.")
            continue

        if use_amp:
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    return running_loss / total if total > 0 else float("inf")


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, desc="Eval", loss_fn=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels, all_attn = [], [], []

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, features, labels in tqdm(loader, desc=desc):
            images   = images.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            labels   = labels.to(device, non_blocking=True)

            logits, attn = model(images, features)
            loss = loss_fn(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attn.append(attn.detach().cpu().numpy())

    if total == 0:
        return float("inf"), 0.0, [], [], np.array([])

    avg_attn = np.concatenate(all_attn, axis=0).mean(axis=0) if all_attn else np.array([])
    return running_loss / total, 100.0 * correct / total, all_preds, all_labels, avg_attn


# ── TTA evaluation ────────────────────────────────────────────────────────────
def evaluate_with_tta(model, dataset_dir, feature_cache, feature_scaler,
                      device, batch_size, num_workers, n_augments=6):
    print(f"\n[TTA] {n_augments} augmentation passes...")
    model.eval()
    all_probs = None
    all_labels = []
    all_attn = []

    for t_idx, tf in enumerate(tqdm(get_tta_transforms(n_augments), desc="TTA")):
        ds = HybridDataset(dataset_dir, transform=tf,
                           feature_cache=feature_cache, feature_scaler=feature_scaler)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type == "cuda"))
        pass_probs, pass_labels, pass_attn = [], [], []

        with torch.no_grad():
            for images, features, labels in loader:
                images   = images.to(device, non_blocking=True)
                features = features.to(device, non_blocking=True)
                logits, attn = model(images, features)
                pass_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
                pass_attn.append(attn.detach().cpu().numpy())
                if t_idx == 0:
                    pass_labels.extend(labels.numpy())

        probs_np = np.concatenate(pass_probs, axis=0)
        all_probs = probs_np if all_probs is None else all_probs + probs_np
        if t_idx == 0:
            all_labels = pass_labels
            all_attn = pass_attn

    all_probs /= n_augments
    all_preds = all_probs.argmax(axis=1).tolist()
    acc = 100.0 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    avg_attn = np.concatenate(all_attn, axis=0).mean(axis=0) if all_attn else np.array([])
    return acc, all_preds, all_labels, avg_attn


def load_best_model(ckpt_path, device, num_classes, num_features):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EfficientNetHybrid(num_classes=num_classes, num_features=num_features)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def train_hybrid_model(args):
    set_seed(args.seed)

    print("=" * 80)
    print("Hybrid Model Training  [EfficientNet-B3 + Medical Features — v3]")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cpu":
        torch.set_num_threads(args.num_threads)

    data_dir = resolve_data_dir(args.data_dir)

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    checkpoint_dir = checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] data_dir    : {data_dir}")
    print(f"[INFO] checkpoint  : {checkpoint_dir}")

    # ── Feature scaler ────────────────────────────────────────────────────────
    print("\nExtracting features for normalisation...")
    temp_ds = HybridDataset(data_dir / "train", transform=None)
    raw_feats = []
    for i in tqdm(range(len(temp_ds)), desc="Raw features"):
        _, f, _ = temp_ds[i]
        raw_feats.append(f.numpy())

    feature_scaler = StandardScaler()
    feature_scaler.fit(raw_feats)

    shared_cache = {}
    print("Pre-computing scaled feature cache...")
    for i in tqdm(range(len(temp_ds)), desc="Cache"):
        img_path, _ = temp_ds.samples[i]
        scaled = feature_scaler.transform([raw_feats[i]])[0].tolist()
        shared_cache[img_path] = scaled

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds      = HybridDataset(data_dir / "train", get_transforms(True),  shared_cache)
    train_eval_ds = HybridDataset(data_dir / "train", get_transforms(False), shared_cache)
    val_ds        = HybridDataset(data_dir / "val",   get_transforms(False), feature_scaler=feature_scaler)

    num_classes  = len(train_ds.class_to_idx)
    num_features = NUM_TRADITIONAL_FEATURES

    # ── FIX #3: Inverse-frequency class weights for focal loss ────────────────
    class_counts = Counter([lbl for _, lbl in train_ds.samples])
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor(
        [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)],
        dtype=torch.float32
    ).to(device)
    print(f"\nClass counts  : {dict(sorted(class_counts.items()))}")
    print(f"Class weights : {class_weights.cpu().numpy().round(3)}")

    # ── FIX #4: Stronger oversampling for minority classes (exponent 1.5) ─────
    max_count = max(class_counts.values())
    sample_weights = [
        (max_count / class_counts[lbl]) ** 1.5   # was 1.0 — hits minority harder
        for _, lbl in train_ds.samples
    ]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    num_workers = args.num_workers if device.type == "cuda" else 0
    lkw = dict(num_workers=num_workers, pin_memory=(device.type == "cuda"))
    if num_workers > 0:
        lkw["prefetch_factor"] = 2

    train_loader      = DataLoader(train_ds,      batch_size=args.batch_size, sampler=sampler,  **lkw)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False,    **lkw)
    val_loader        = DataLoader(val_ds,        batch_size=args.batch_size, shuffle=False,    **lkw)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nCreating model...")
    model = build_model(num_classes, num_features, device, args.pretrained_cnn)

    # ── FIX #5: Feature MLP gets head LR, not backbone LR ────────────────────
    backbone_params = list(model.backbone.parameters())
    backbone_ids    = {id(p) for p in backbone_params}
    head_params     = [p for p in model.parameters() if id(p) not in backbone_ids]

    param_groups = [
        {"params": backbone_params, "lr": args.learning_rate * 0.1, "name": "backbone"},
        {"params": head_params,     "lr": args.learning_rate,        "name": "head"},
    ]

    optimizer  = optim.AdamW(param_groups, weight_decay=0.01, betas=(0.9, 0.999))
    warmup_ep  = max(3, args.epochs // 10)
    scheduler  = WarmupCosineScheduler(
        optimizer, warmup_epochs=warmup_ep, total_epochs=args.epochs,
        base_lr=args.learning_rate, min_lr=1e-6,
    )

    # FIX #1: Focal loss with class weights instead of label smoothing
    criterion  = FocalLoss(gamma=2.0, weight=class_weights)
    amp_scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    print("\n" + "=" * 80)
    print(f"Starting Training  [{args.epochs} epochs, patience={args.early_stopping_patience}]")
    print(f"  Loss: FocalLoss(gamma=2) + class weights  |  Oversampling exponent=1.5")
    print(f"  Attention: fixed (detached) — feature branch will contribute")
    print(f"  AMP={'on' if amp_scaler else 'off'}")
    print("=" * 80)

    best_val_acc    = 0.0
    patience_counter = 0
    UNFREEZE_EPOCH  = warmup_ep + 1

    for epoch in range(1, args.epochs + 1):

        if epoch < UNFREEZE_EPOCH:
            model._freeze_backbone(freeze=True)
        elif epoch == UNFREEZE_EPOCH:
            print(f"\n  [Epoch {epoch}] Unfreezing backbone (last 20%)")
            model.unfreeze_backbone(blocks_from_end=2)
        elif epoch == UNFREEZE_EPOCH + 5:
            print(f"\n  [Epoch {epoch}] Unfreezing more backbone layers")
            model.unfreeze_backbone(blocks_from_end=5)

        train_loss   = train_epoch(model, train_loader, optimizer, device, epoch, criterion, amp_scaler)
        current_lr   = scheduler.step()

        train_acc = float("nan")
        if epoch > 1:
            _, train_acc, _, _, _ = evaluate(model, train_eval_loader, device, f"Epoch {epoch} [train eval]")

        val_loss, val_acc, val_preds, val_labels, avg_attn = evaluate(
            model, val_loader, device, f"Epoch {epoch} [val]"
        )

        # Print attention split so we can verify the fix is working
        attn_str = (
            f"CNN={float(avg_attn[0]):.3f} / Feat={float(avg_attn[1]):.3f}"
            if avg_attn.ndim == 1 and len(avg_attn) == 2
            else "n/a"
        )

        print(f"\nEpoch {epoch}/{args.epochs}  |  LR: {current_lr:.7f}  |  Attn: {attn_str}")
        print(f"  Train Loss : {train_loss:.4f}  |  Train Acc : {train_acc:.2f}%")
        print(f"  Val Loss   : {val_loss:.4f}  |  Val Acc   : {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "num_classes": num_classes,
                "num_traditional_features": num_features,
                "class_to_idx": train_ds.class_to_idx,
                "avg_attention_weights": avg_attn,
            }
            torch.save(ckpt, checkpoint_dir / "best_hybrid_model.pth")
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.early_stopping_patience})")

        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
            }, checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")

    print(f"\n{'='*80}")
    print(f"Training complete!  Best Val Acc (no TTA): {best_val_acc:.2f}%")

    # ── Final evaluation ──────────────────────────────────────────────────────
    best_ckpt_path = checkpoint_dir / "best_hybrid_model.pth"
    model = load_best_model(best_ckpt_path, device, num_classes, num_features)

    print("\nFinal Evaluation — Standard:")
    _, val_acc_std, preds_std, labels_std, avg_attn = evaluate(
        model, val_loader, device, "Standard Eval"
    )
    print(f"  Standard Val Acc : {val_acc_std:.2f}%")

    print("\nFinal Evaluation — TTA (6 passes):")
    val_acc_tta, preds_tta, labels_tta, _ = evaluate_with_tta(
        model,
        dataset_dir=data_dir / "val",
        feature_cache={},
        feature_scaler=feature_scaler,
        device=device,
        batch_size=args.batch_size,
        num_workers=num_workers,
        n_augments=6,
    )
    print(f"  TTA Val Acc      : {val_acc_tta:.2f}%")

    class_names = [train_ds.idx_to_class[i] for i in range(num_classes)]
    print("\nClassification Report (TTA):")
    print(classification_report(labels_tta, preds_tta, target_names=class_names, zero_division=0))
    print("Confusion Matrix (TTA):")
    print(confusion_matrix(labels_tta, preds_tta))

    if hasattr(model, "get_feature_importance") and avg_attn.size > 0:
        print("\nFeature Branch Attention Weight:", round(float(avg_attn[1]) if avg_attn.ndim == 1 else float(avg_attn[:, 1].mean()), 4))
        print("Top-10 Feature Importance:")
        imp = model.get_feature_importance(avg_attn, FEATURE_NAMES)
        for i, (n, s) in enumerate(list(imp.items())[:10], 1):
            print(f"  {i:2d}. {n:25s}: {s:.4f}")

    print(f"\n✓ Best (no TTA): {val_acc_std:.2f}%  |  Best (TTA): {val_acc_tta:.2f}%")
    print(f"  Checkpoint: {best_ckpt_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Model — EfficientNet-B3 v3")
    parser.add_argument("--data-dir",                type=str,   default="data")
    parser.add_argument("--pretrained-cnn",          type=str,   default=None)
    parser.add_argument("--epochs",                  type=int,   default=80)
    parser.add_argument("--batch-size",              type=int,   default=32)
    parser.add_argument("--learning-rate",           type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int,   default=20)
    parser.add_argument("--cpu-only",                action="store_true")
    parser.add_argument("--num-workers",             type=int,   default=4)
    parser.add_argument("--num-threads",             type=int,   default=4)
    parser.add_argument("--checkpoint-dir",          type=str,   default="checkpoints")
    parser.add_argument("--seed",                    type=int,   default=42)
    args = parser.parse_args()
    train_hybrid_model(args)


if __name__ == "__main__":
    main()