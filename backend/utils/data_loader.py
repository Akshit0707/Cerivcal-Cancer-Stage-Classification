"""
dataset.py — aligned with train_hybrid.py v11
(SIPaKMeD + Herlev | 4-class ordinal)

Key changes vs original:
  - 4-class ordinal label space : Normal → CIN1 → HighGrade → Cancer
  - NUM_FEATURES = 31            : 30 medical dims + 1 synthetic flag
  - INPUT_SIZE   = 300           : matches train_hybrid.py
  - HybridDataset returns dicts  : {image, features, label, path}
  - get_transforms / tta_transforms mirror train_hybrid.py exactly
  - get_data_loaders supports WeightedRandomSampler + feature cache
  - extract_medical_features + RobustScaler wiring kept compatible
"""

import os
import random
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

warnings.filterwarnings("ignore")

# ── Constants (must match train_hybrid.py) ───────────────────────────────────

SEVERITY_ORDER = ['Normal', 'CIN1', 'HighGrade', 'Cancer']   # ordinal order
HARD_CLASSES   = {'HighGrade', 'Cancer', 'Normal', 'CIN1'}   # upsampled classes
NUM_FEATURES   = 31    # 30 medical dims + 1 synthetic flag
INPUT_SIZE     = 300   # EfficientNet-B3 native resolution


# ── Medical feature extraction ───────────────────────────────────────────────

def _load_feature_extractor():
    """Import extract_medical_features from wherever it lives."""
    import sys
    for search in (Path(__file__).resolve().parent,
                   Path(__file__).resolve().parents[1]):
        if str(search) not in sys.path:
            sys.path.insert(0, str(search))
    try:
        from backend.feature_extractor import extract_medical_features
        return extract_medical_features
    except ImportError:
        pass
    try:
        from feature_extractor import extract_medical_features
        return extract_medical_features
    except ImportError as e:
        raise ImportError("Could not import extract_medical_features.") from e


def _sanitize(arr: np.ndarray) -> np.ndarray:
    return np.clip(
        np.nan_to_num(np.array(arr, np.float32), nan=0., posinf=0., neginf=0.),
        -1e6, 1e6,
    )


def build_feature_cache(
    paths: list,
    scaler: RobustScaler = None,
    fit_scaler: bool = False,
) -> tuple:
    """
    Extract 30-dim medical features for every path and optionally fit/apply
    a RobustScaler.  Returns (cache_dict, scaler).

    cache_dict maps path -> np.ndarray of shape (30,)  [scaled if scaler given]
    The 31st synthetic-flag dimension is appended in HybridDataset.__getitem__.
    """
    extract = _load_feature_extractor()
    N_MED   = NUM_FEATURES - 1          # 30
    raw     = {}
    n_bad   = 0

    for p in tqdm(paths, desc="Extracting features", leave=False):
        try:
            f = extract(Image.open(p).convert('RGB'))
            f = _sanitize(f)
            if len(f) != N_MED:
                raise ValueError(f"expected {N_MED} features, got {len(f)}")
        except Exception:
            f = np.zeros(N_MED, np.float32)
            n_bad += 1
        raw[p] = f

    if n_bad:
        pct = 100 * n_bad / max(1, len(paths))
        print(f"  ⚠️  {n_bad} ({pct:.1f}%) feature failures — zeros used")
        if pct > 30:
            print("  ❌  >30% failures — check feature_extractor!")

    # Sanity check
    sample_keys = random.sample(list(raw), min(50, len(raw)))
    sample = np.concatenate([raw[k] for k in sample_keys])
    if np.allclose(sample, 0.):
        print("  ❌  CRITICAL: all features are zero!")
    else:
        print(f"  ✅  Feature range [{sample.min():.3f}, {sample.max():.3f}]")

    if fit_scaler:
        scaler = RobustScaler(quantile_range=(10., 90.))
        scaler.fit(np.stack([raw[p] for p in paths]))

    cache = {}
    for p in paths:
        if scaler is not None:
            cache[p] = _sanitize(np.clip(scaler.transform([raw[p]])[0], -3., 3.))
        else:
            cache[p] = raw[p]

    return cache, scaler


# ── Dataset ───────────────────────────────────────────────────────────────────

class HybridDataset(Dataset):
    """
    Returns dicts compatible with train_hybrid.py's train_epoch / evaluate:

        {
          'image'   : FloatTensor [3, H, W],
          'features': FloatTensor [31],   # 30 medical + 1 synthetic flag
          'label'   : LongTensor  scalar,
          'path'    : str,
        }

    Args:
        paths     : list of image file paths (str)
        labels    : list of int class indices aligned with SEVERITY_ORDER
        transform : torchvision transform applied to PIL image
        cache     : dict {path -> np.ndarray shape (30,)} or None
                    When None, the 30 medical features are all zeros.
    """

    def __init__(
        self,
        paths: list,
        labels: list,
        transform=None,
        cache: dict = None,
    ):
        assert len(paths) == len(labels), "paths/labels length mismatch"
        self.paths     = paths
        self.labels    = labels
        self.transform = transform
        self.cache     = cache or {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        p   = self.paths[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # 30 medical dims (zeros if not cached) + 1 synthetic flag
        med_feat = self.cache.get(p, np.zeros(NUM_FEATURES - 1, dtype=np.float32))
        is_syn   = 1.0 if 'syn_' in Path(p).name else 0.0
        feat     = np.append(med_feat, is_syn).astype(np.float32)

        return {
            'image':    img,
            'features': torch.FloatTensor(feat),
            'label':    torch.tensor(self.labels[idx], dtype=torch.long),
            'path':     p,
        }


# ── Transforms (mirrors train_hybrid.py exactly) ─────────────────────────────

def get_train_transform(sz: int = INPUT_SIZE) -> transforms.Compose:
    """Heavy augmentation used during training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            sz, scale=(0.7, 1.0), ratio=(0.85, 1.15),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.06),
        transforms.RandomAffine(0, translate=(0.1, 0.1),
                                scale=(0.9, 1.1), shear=8),
        transforms.RandomGrayscale(p=0.06),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


def get_val_transform(sz: int = INPUT_SIZE) -> transforms.Compose:
    """Deterministic transform for validation / inference."""
    return transforms.Compose([
        transforms.Resize((sz, sz),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_tta_transforms(sz: int = INPUT_SIZE) -> list:
    """
    5-view TTA list used in evaluate() — identical to train_hybrid.py.
    Index 0 is the base (no-aug) transform; indices 1-4 are flips/rotations.
    """
    n = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    base = get_val_transform(sz)
    return [
        base,                                                                   # 0: base
        transforms.Compose([transforms.RandomHorizontalFlip(1.),               # 1: H-flip
                             transforms.Resize((sz, sz), interpolation=transforms.InterpolationMode.BICUBIC),
                             transforms.ToTensor(), n]),
        transforms.Compose([transforms.RandomVerticalFlip(1.),                 # 2: V-flip
                             transforms.Resize((sz, sz), interpolation=transforms.InterpolationMode.BICUBIC),
                             transforms.ToTensor(), n]),
        transforms.Compose([transforms.RandomRotation((90, 90)),               # 3: rot90
                             transforms.Resize((sz, sz), interpolation=transforms.InterpolationMode.BICUBIC),
                             transforms.ToTensor(), n]),
        transforms.Compose([transforms.RandomRotation((180, 180)),             # 4: rot180
                             transforms.Resize((sz, sz), interpolation=transforms.InterpolationMode.BICUBIC),
                             transforms.ToTensor(), n]),
    ]


# ── Weighted sampler ──────────────────────────────────────────────────────────

def make_weighted_sampler(labels: list, class_names: list) -> WeightedRandomSampler:
    """
    Mirrors train_hybrid.py sampler weights:
      CIN1      → 5× inverse-frequency boost
      HighGrade / Cancer / Normal → 3× inverse-frequency boost
      others    → 1× inverse-frequency
    """
    tc = Counter(labels)

    def _weight(lbl: int) -> float:
        name = class_names[lbl]
        base = 1.0 / max(1, tc[lbl])
        if name == 'CIN1':
            return base * 5.0
        if name in HARD_CLASSES:
            return base * 3.0
        return base

    sample_weights = [_weight(l) for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels),
                                 replacement=True)


# ── Image discovery ───────────────────────────────────────────────────────────

_IMG_EXTS = ('*.jpg', '*.JPG', '*.jpeg', '*.png', '*.PNG', '*.bmp', '*.BMP')


def _glob_images(directory: Path) -> list:
    imgs = []
    for pat in _IMG_EXTS:
        imgs.extend(sorted(directory.glob(pat)))
    return imgs


def _load_split(split_dir: Path, class_names: list) -> tuple:
    """Scan split_dir/<class>/ and return (paths, labels)."""
    paths, labels = [], []
    for idx, cls in enumerate(class_names):
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue
        imgs = _glob_images(cls_dir)
        print(f"     {cls}: {len(imgs)}")
        for img in imgs:
            paths.append(str(img))
            labels.append(idx)
    return paths, labels


# ── Main entry point ──────────────────────────────────────────────────────────

def get_data_loaders(
    data_dir:    str,
    batch_size:  int  = 32,
    num_workers: int  = 4,
    use_cache:   bool = True,
    use_sampler: bool = True,
    val_size:    float = 0.2,
    seed:        int  = 42,
) -> tuple:
    """
    Build train / val DataLoaders from data_dir.

    Supported layouts
    -----------------
    A) Pre-split  :  data_dir/train/<class>/  and  data_dir/val/<class>/
    B) Flat       :  data_dir/<class>/  (auto-split 80/20 stratified)

    Returns
    -------
    (train_loader, val_loader, class_names, scaler)
        scaler is a fitted RobustScaler when use_cache=True, else None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dp   = Path(data_dir)
    trp  = dp / 'train'
    vap  = dp / 'val'

    # ── Discover class names ────────────────────────────────────────────────
    if trp.exists() and vap.exists():
        cls = sorted([p.name for p in trp.iterdir() if p.is_dir()])
        print(f"✅ Pre-split layout detected. Classes: {cls}")
        print("📂 train/");  tr_paths, tr_labels = _load_split(trp, cls)
        print("📂 val/");    va_paths, va_labels = _load_split(vap, cls)

        # Re-split if val is severely imbalanced
        vc = Counter(va_labels)
        if len(vc) > 0 and max(vc.values()) / max(min(vc.values()), 1) > 3:
            print("⚠️  Val is severely imbalanced — re-splitting 80/20 stratified")
            all_p = tr_paths + va_paths
            all_l = tr_labels + va_labels
            tr_paths, va_paths, tr_labels, va_labels = train_test_split(
                all_p, all_l,
                test_size=val_size, random_state=seed, stratify=all_l,
            )
    else:
        cls = sorted([
            p.name for p in dp.iterdir()
            if p.is_dir() and p.name not in
            ('test', 'synthetic', 'sipakmed_raw', '__pycache__')
        ])
        print(f"✅ Flat layout detected. Classes: {cls}")
        all_p, all_l = _load_split(dp, cls)
        tr_paths, va_paths, tr_labels, va_labels = train_test_split(
            all_p, all_l,
            test_size=val_size, random_state=seed, stratify=all_l,
        )

    if not tr_paths:
        raise ValueError(f"No training images found under: {data_dir}")

    print(f"\n✅ Train: {len(tr_paths)}  Val: {len(va_paths)}")

    # ── Remap labels to SEVERITY_ORDER if all classes are known ────────────
    sev_present = [c for c in SEVERITY_ORDER if c in cls]
    if set(sev_present) == set(cls):
        old2new  = {cls.index(c): sev_present.index(c) for c in cls}
        tr_labels = [old2new[l] for l in tr_labels]
        va_labels = [old2new[l] for l in va_labels]
        cls       = sev_present
        print(f"✅ Labels remapped to severity order: {cls}")
    else:
        unknown = set(cls) - set(SEVERITY_ORDER)
        if unknown:
            print(f"⚠️  Unknown classes {unknown} — ordinal loss may be less effective.")

    # ── Feature extraction ──────────────────────────────────────────────────
    tr_cache = va_cache = scaler = None
    if use_cache:
        print("\n🔍 Extracting train features (30 medical dims)...")
        tr_cache, scaler = build_feature_cache(tr_paths, fit_scaler=True)
        print("🔍 Extracting val features...")
        va_cache, _      = build_feature_cache(va_paths, scaler=scaler)

    # ── Datasets ────────────────────────────────────────────────────────────
    tr_ds = HybridDataset(tr_paths, tr_labels,
                          transform=get_train_transform(), cache=tr_cache)
    va_ds = HybridDataset(va_paths, va_labels,
                          transform=get_val_transform(),   cache=va_cache)

    # ── Sampler ─────────────────────────────────────────────────────────────
    sampler = make_weighted_sampler(tr_labels, cls) if use_sampler else None

    nw = min(num_workers, os.cpu_count() or 0)
    tr_loader = DataLoader(
        tr_ds,
        batch_size      = batch_size,
        sampler         = sampler,          # mutually exclusive with shuffle
        shuffle         = (sampler is None),
        num_workers     = nw,
        pin_memory      = torch.cuda.is_available(),
        drop_last       = True,
        persistent_workers = (nw > 0),
    )
    va_loader = DataLoader(
        va_ds,
        batch_size      = batch_size,
        shuffle         = False,
        num_workers     = nw,
        pin_memory      = torch.cuda.is_available(),
        persistent_workers = (nw > 0),
    )

    # Per-class summary
    tc, vc2 = Counter(tr_labels), Counter(va_labels)
    print("\n  Class distribution:")
    for i, name in enumerate(cls):
        syn_tr = sum(1 for p in tr_paths
                     if tr_labels[tr_paths.index(p)] == i and 'syn_' in Path(p).name)
        print(f"    {name:12s}: train={tc[i]:4d}  val={vc2[i]:4d}"
              f"  (syn_train={syn_tr})")

    return tr_loader, va_loader, cls, scaler


# ── Backward-compat alias for scripts that call get_transforms() ─────────────

def get_transforms(augment: bool = True) -> transforms.Compose:
    """Legacy alias — prefer get_train_transform() / get_val_transform()."""
    return get_train_transform() if augment else get_val_transform()


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', nargs='?', default='./data',
                        help='Root data directory')
    parser.add_argument('--batch-size',  type=int, default=4)
    parser.add_argument('--no-cache',    action='store_true')
    parser.add_argument('--no-sampler',  action='store_true')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"dataset.py smoke-test  |  INPUT_SIZE={INPUT_SIZE}  NUM_FEATURES={NUM_FEATURES}")
    print(f"SEVERITY_ORDER: {SEVERITY_ORDER}")
    print(f"{'='*60}\n")

    try:
        tr_loader, va_loader, cls, scaler = get_data_loaders(
            args.data_dir,
            batch_size  = args.batch_size,
            num_workers = 0,
            use_cache   = not args.no_cache,
            use_sampler = not args.no_sampler,
        )
    except ValueError as e:
        print(f"\n⚠️  {e}")
        print("Create dummy data? Run with a populated data_dir.")
        sys.exit(0)

    print(f"\n✅ Classes: {cls}")
    print(f"   train batches: {len(tr_loader)}  val batches: {len(va_loader)}")

    batch = next(iter(tr_loader))
    print(f"\nSample batch:")
    print(f"  image   : {batch['image'].shape}   dtype={batch['image'].dtype}")
    print(f"  features: {batch['features'].shape} dtype={batch['features'].dtype}")
    print(f"  label   : {batch['label'].shape}    values={batch['label'].tolist()}")
    print(f"  path[0] : {batch['path'][0]}")

    feat = batch['features']
    print(f"\nFeature stats: min={feat.min():.3f}  max={feat.max():.3f}"
          f"  mean={feat.mean():.3f}  std={feat.std():.3f}")
    print(f"Synthetic flag (last dim): {feat[:, -1].tolist()}")
    print("\n✅ Smoke test passed.")