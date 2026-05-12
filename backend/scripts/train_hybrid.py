"""
Training Script for GPU-Optimized Hybrid Model
Combines CNN features with traditional medical features
"""

import argparse
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
BACKEND_DIR = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

for p in (SCRIPTS_DIR, BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from feature_extractor import extract_medical_features
from models.hybrid_model import create_hybrid_model, load_hybrid_model

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

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
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
            features = extract_medical_features(image)
            self.feature_cache[img_path] = features

        if self.transform is not None:
            image = self.transform(image)

        if self.feature_scaler is not None:
            features = self.feature_scaler.transform([features])[0]

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image, features_tensor, label_tensor


# ---------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------
def get_transforms(augment=True):
    if augment:
        return transforms.Compose(
            [
                # use RandomResizedCrop to increase effective resolution + scale variability
                transforms.RandomResizedCrop(320, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.08, 0.08),
                    scale=(0.9, 1.1),
                    shear=6,
                ),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.06
                ),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# ---------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam, criterion):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------
def train_epoch(model, train_loader, optimizer, device, epoch, criterion, scaler=None):
    model.train()
    running_loss = 0.0
    total = 0
    use_amp = device.type == "cuda" and scaler is not None

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, features, labels in pbar:
        images = images.to(device, non_blocking=True)
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        images_mixed, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits, _ = model(images_mixed, features)
                loss = mixup_criterion(logits, labels_a, labels_b, lam, criterion)
        else:
            logits, _ = model(images_mixed, features)
            loss = mixup_criterion(logits, labels_a, labels_b, lam, criterion)

        if total == 0:
            print(f"\nDEBUG - Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
            print(f"DEBUG - Mixup lambda: {lam:.2f}")
            if use_amp:
                print("DEBUG - Using AMP")

        if torch.isnan(loss) or torch.isinf(loss):
            print("\nWARNING: Invalid loss encountered. Skipping batch.")
            continue

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

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if total == 0:
        return float("inf")

    return running_loss / total


def evaluate(model, loader, device, desc="Evaluating", loss_fn=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds, all_labels, all_attention_weights = [], [], []

    if loss_fn is None:
        loss_fn = F.cross_entropy

    with torch.no_grad():
        for images, features, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, attention_weights = model(images, features)
            loss = loss_fn(logits, labels)

            running_loss += loss.item() * images.size(0)
            predicted = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attention_weights.append(attention_weights.detach().cpu().numpy())

    if total == 0:
        return float("inf"), 0.0, all_preds, all_labels, np.array([])

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    avg_attention = (
        np.concatenate(all_attention_weights, axis=0).mean(axis=0)
        if all_attention_weights
        else np.array([])
    )
    return epoch_loss, epoch_acc, all_preds, all_labels, avg_attention


def load_best_model(checkpoint_path: Path, device, num_classes: int, num_features: int):
    try:
        return load_hybrid_model(checkpoint_path, device=device)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = create_hybrid_model(
            num_classes=num_classes,
            num_traditional_features=num_features,
            pretrained_cnn_path=None,
            device=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------
def train_hybrid_model(args):
    set_seed(args.seed)

    print("=" * 80)
    print("Hybrid Model Training")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu")
    print(f"\nUsing device: {device}")

    if device.type == "cpu":
        torch.set_num_threads(args.num_threads)

    data_dir = resolve_data_dir(args.data_dir)
    checkpoint_dir = (PROJECT_ROOT / args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] project_root: {PROJECT_ROOT}")
    print(f"[INFO] data_dir: {data_dir}")
    print(f"[INFO] checkpoint_dir: {checkpoint_dir}")

    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)

    print("Extracting features for normalization...")
    temp_dataset = HybridDataset(data_dir / "train", transform=None)

    all_features = []
    for i in tqdm(range(len(temp_dataset)), desc="Collecting features"):
        _, features, _ = temp_dataset[i]
        all_features.append(features.numpy())

    feature_scaler = StandardScaler()
    feature_scaler.fit(all_features)
    print(
        f"Feature normalization fitted (first3 mean={feature_scaler.mean_[:3]}, "
        f"std={feature_scaler.scale_[:3]})"
    )

    train_dataset = HybridDataset(
        data_dir / "train",
        transform=train_transform,
        feature_scaler=feature_scaler,
    )
    train_eval_dataset = HybridDataset(
        data_dir / "train",
        transform=val_transform,
        feature_scaler=feature_scaler,
    )
    val_dataset = HybridDataset(
        data_dir / "val",
        transform=val_transform,
        feature_scaler=feature_scaler,
    )

    class_counts = Counter([label for _, label in train_dataset.samples])

    print("\nClass distribution:")
    for cls_idx in sorted(class_counts.keys()):
        print(f"  {train_dataset.idx_to_class[cls_idx]}: {class_counts[cls_idx]}")

    max_count = max(class_counts.values())
    sample_weights = [(max_count / class_counts[label]) ** 1.0 for _, label in train_dataset.samples]

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    num_workers = args.num_workers if device.type == "cuda" else 0
    train_loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    train_eval_loader_kwargs = dict(
        dataset=train_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader_kwargs = dict(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2
        train_eval_loader_kwargs["prefetch_factor"] = 2
        val_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(**train_loader_kwargs)
    train_eval_loader = DataLoader(**train_eval_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)

    print("\nCreating model...")
    num_classes = len(train_dataset.class_to_idx)
    num_features = NUM_TRADITIONAL_FEATURES

    model = create_hybrid_model(
        num_classes=num_classes,
        num_traditional_features=num_features,
        pretrained_cnn_path=args.pretrained_cnn,
        device=device,
    )

    class_weights = []
    for cls_idx in sorted(class_counts.keys()):
        weight = (max_count / class_counts[cls_idx]) ** 1.0
        class_weights.append(weight)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    print(f"\nClass weights: {class_weights.detach().cpu().numpy()}")
    print("Using Focal Loss (gamma=2.5)")
    criterion = FocalLoss(alpha=class_weights, gamma=2.5)

    lr = args.learning_rate * 2.0 if device.type == "cuda" else args.learning_rate
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    print("\n" + "=" * 80)
    print("Starting Training")
    if device.type == "cuda":
        print(f"✓ GPU Training with AMP enabled (LR: {lr:.6f})")
    print("=" * 80)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, criterion, scaler
        )
        train_eval_loss, train_eval_acc, _, _, _ = evaluate(
            model, train_eval_loader, device, desc="Train Eval"
        )
        val_loss, val_acc, val_preds, val_labels, avg_attention = evaluate(
            model, val_loader, device, desc="Validating"
        )

        scheduler.step(val_acc)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Acc: {train_eval_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "num_classes": num_classes,
                "num_traditional_features": num_features,
                "class_to_idx": train_dataset.class_to_idx,
                "avg_attention_weights": avg_attention,
            }

            checkpoint_path = checkpoint_dir / "best_hybrid_model.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.early_stopping_patience})")

        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

        if epoch % 10 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                },
                checkpoint_path,
            )
            print(f"  Saved checkpoint at epoch {epoch}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")

    best_checkpoint_path = checkpoint_dir / "best_hybrid_model.pth"
    model = load_best_model(
        best_checkpoint_path,
        device=device,
        num_classes=num_classes,
        num_features=num_features,
    )

    print("\nFinal Evaluation on Validation Set:")
    val_loss, val_acc, val_preds, val_labels, avg_attention = evaluate(
        model, val_loader, device, desc="Validating"
    )

    class_names = [train_dataset.idx_to_class[i] for i in range(num_classes)]
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=class_names, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(val_labels, val_preds)
    print(cm)

    print("\n" + "=" * 80)
    print("Feature Importance Analysis")
    print("=" * 80)

    if hasattr(model, "get_feature_importance") and avg_attention.size > 0:
        importance_dict = model.get_feature_importance(avg_attention, FEATURE_NAMES)
        print("\nTop 10 Most Important Features:")
        for i, (name, score) in enumerate(list(importance_dict.items())[:10], 1):
            print(f"  {i:2d}. {name:25s}: {score:.4f}")

    print("\n✓ Training completed successfully!")
    print(f"Best model saved at: {best_checkpoint_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Model")

    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument(
        "--pretrained-cnn",
        type=str,
        default=None,
        help="Path to pretrained CNN model (optional)",
    )

    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="Base learning rate")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=25,
        help="Early stopping patience",
    )

    parser.add_argument("--cpu-only", action="store_true", help="Force CPU training")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loading workers")
    parser.add_argument("--num-threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train_hybrid_model(args)


if __name__ == "__main__":
    main()
