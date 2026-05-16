"""
train_hybrid.py — v11 (SIPaKMeD + Herlev | 4-class ordinal)
═══════════════════════════════════════════════════════════════════

DATASET: SIPaKMeD real Pap smear cells + Herlev carcinoma-in-situ
         + optional img2img synthetic augmentation

4 CLASSES (severity order):
  0 — Normal     : SIPaKMeD Superficial-Intermediate + Parabasal + Metaplastic
  1 — CIN1       : SIPaKMeD Koilocytotic
  2 — HighGrade  : SIPaKMeD Dyskeratotic  (CIN2 + CIN3 merged)
  3 — Cancer     : Herlev carcinoma_in_situ

KEY CHANGES FROM v10:
  - SEVERITY_ORDER updated to 4-class scheme
  - NUM_FEATURES = 31  (30 medical + 1 is_synthetic flag)
  - Sampler: HighGrade + Cancer get 3x weight (Normal is now abundant)
  - Hard-class list updated: HighGrade, Cancer
  - Data loader: accepts flat folder OR train/val split
  - Synthetic flag injected into feature vector automatically
  - SWA_START lowered to 45 (more data → faster convergence)
  - Checkpoint keys updated (saves class_names correctly)
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
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_VERSION = "EfficientNet-B3-Hybrid-v11-sipakmed-4class"
print(f"[train_hybrid.py] version={_VERSION}  file={__file__}")

# ─────────────────────────────────────────────────────────────────────────────
# Toggles
# ─────────────────────────────────────────────────────────────────────────────
USE_SAM    = False
USE_TTA    = True
USE_SWA    = True
USE_CUTMIX = True
SWA_START  = 45          # lowered from 55 — more data converges faster
TTA_EVERY  = 5

# ─────────────────────────────────────────────────────────────────────────────
# Architecture constants
# ─────────────────────────────────────────────────────────────────────────────
NUM_FEATURES    = 31     # 30 medical features + 1 is_synthetic flag
INPUT_SIZE      = 300
FEAT_DIM        = 128
CNN_DIM         = 1536
FUSION_DIM      = CNN_DIM + FEAT_DIM   # 1664

FREEZE_EPOCHS   = 10
UNFREEZE_STEP   = 20
ORDINAL_WEIGHT  = 0.3
CE_WEIGHT       = 0.7

# ── 4-class severity order ────────────────────────────────────────────────────
# MUST match your folder names exactly (case-sensitive)
SEVERITY_ORDER = ['Normal', 'CIN1', 'HighGrade', 'Cancer']

# Classes that get 3x sampler weight (harder / less represented)
HARD_CLASSES   = {'HighGrade', 'Cancer', 'Normal', 'CIN1'}  # In 4-class setup, all classes are somewhat hard/imbalanced

# ─────────────────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_PATH = Path(__file__).resolve()
for _p in (SCRIPT_PATH.parent, SCRIPT_PATH.parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from backend.feature_extractor import extract_medical_features
except Exception:
    try:
        from feature_extractor import extract_medical_features
    except Exception as _e:
        raise ImportError("Could not import feature_extractor.") from _e


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────────────────────
# ORDINAL LOSS
# ─────────────────────────────────────────────────────────────────────────────
class OrdinalLoss(nn.Module):
    def __init__(self, num_classes=4, smoothing=0.05):
        super().__init__()
        self.K = num_classes
        self.s = smoothing

    def forward(self, logits, targets):
        B, K1 = logits.shape
        bt = torch.zeros(B, K1, device=targets.device)
        for k in range(1, self.K):
            bt[:, k-1] = (targets >= k).float()
        bt = bt*(1-self.s) + (1-bt)*self.s
        return F.binary_cross_entropy_with_logits(logits, bt)


class FocalLoss(nn.Module):
    """Focal Loss — gamma=1.0 (gentle). Sampler handles class balance."""
    def __init__(self, gamma=1.0, smoothing=0.08, num_classes=4):
        super().__init__()
        self.gamma = gamma
        self.s = smoothing
        self.K = num_classes

    def forward(self, logits, targets):
        logits = torch.clamp(logits, -50., 50.)
        with torch.no_grad():
            sm = torch.full_like(logits, self.s/(self.K-1))
            sm.scatter_(1, targets.unsqueeze(1), 1.-self.s)
        log_p = F.log_softmax(logits, -1)
        ce = -(sm*log_p).sum(-1)
        pt = F.softmax(logits,-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        return ((1-pt)**self.gamma * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class ChannelSE(nn.Module):
    def __init__(self, dim, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(8, dim//r)), nn.ReLU(inplace=True),
            nn.Linear(max(8, dim//r), dim), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x)


class FeatureMLP(nn.Module):
    """3-block MLP with learned gate. Accepts 31-dim input (30 medical + 1 syn flag)."""
    def __init__(self, in_dim=31, out_dim=128, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 256),   nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.Linear(256, out_dim), nn.BatchNorm1d(out_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x) * self.gate(x)


class EfficientNetHybrid(nn.Module):
    """
    EfficientNet-B3 + FeatureMLP (31-dim) with SE attention.
    Two heads:
      - main head:    4-class classifier
      - ordinal head: 3 binary thresholds (Normal<CIN1<HighGrade<Cancer)
    """
    def __init__(self, num_classes=4, num_features=31,
                 feat_dim=128, dropout=0.3, drop_path_rate=0.2):
        super().__init__()
        self.num_classes = num_classes

        try:
            from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
            bb = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.cnn_dim = bb.classifier[1].in_features
            bb.classifier = nn.Identity()
            self._set_drop_path(bb, drop_path_rate)
            self.backbone = bb
        except Exception:
            from torchvision.models import resnet50, ResNet50_Weights
            bb = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.cnn_dim = bb.fc.in_features
            bb.fc = nn.Identity()
            self.backbone = bb

        self.feat_mlp = FeatureMLP(num_features, feat_dim, dropout=0.25)
        fusion = self.cnn_dim + feat_dim
        self.se = ChannelSE(fusion, r=16)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Dropout(dropout*0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(dropout*0.25),
            nn.Linear(256, num_classes),
        )

        self.ordinal_head = nn.Sequential(
            nn.Dropout(dropout*0.5),
            nn.Linear(fusion, 128), nn.GELU(),
            nn.Linear(128, num_classes-1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in list(self.head.modules()) + list(self.ordinal_head.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _set_drop_path(self, bb, rate):
        try:
            blocks = list(bb.features.children())
            n = sum(1 for s in blocks if hasattr(s,'__iter__') for _ in s)
            rates = torch.linspace(0, rate, max(n,1)).tolist()
            i = 0
            for s in blocks:
                if not hasattr(s,'__iter__'): continue
                for b in s:
                    if hasattr(b,'stochastic_depth'):
                        b.stochastic_depth.p = rates[i]
                    i += 1
        except Exception:
            pass

    def forward(self, images, features, return_ordinal=True):
        images   = torch.nan_to_num(images,   nan=0., posinf=1.,  neginf=-1.)
        features = torch.nan_to_num(features, nan=0., posinf=0.,  neginf=0.)
        cnn  = torch.nan_to_num(self.backbone(images),   nan=0., posinf=1e3, neginf=-1e3)
        feat = torch.nan_to_num(self.feat_mlp(features), nan=0., posinf=1e3, neginf=-1e3)
        fused  = self.se(torch.cat([cnn, feat], dim=1))
        logits = self.head(fused)
        if return_ordinal:
            return logits, self.ordinal_head(fused)
        return logits


def build_model(num_classes, num_features, device, dropout=0.3, dpr=0.2):
    m = EfficientNetHybrid(num_classes, num_features, FEAT_DIM, dropout, dpr).to(device)
    n = sum(p.numel() for p in m.parameters())
    t = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  Model      : EfficientNetHybrid-v11 @ {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"  Fusion dim : {FUSION_DIM}  |  Params: {n:,} ({t:,} trainable)")
    print(f"  Device     : {device}  |  Classes: {num_classes}  |  Features: {num_features}")
    return m


def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in model.backbone.parameters())
    print(f"  Backbone frozen ({n:,} params) — head-only for {FREEZE_EPOCHS} epochs")


def get_stages(model):
    try:
        return list(model.backbone.features.children())
    except Exception:
        return []


def unfreeze_progressive(model, epoch, freeze_start):
    stages = get_stages(model)
    if not stages:
        for p in model.backbone.parameters(): p.requires_grad = True
        print("  Backbone fully unfrozen")
        return
    n = min(len(stages), 1 + max(0, (epoch-freeze_start-1)//UNFREEZE_STEP))
    for i, s in enumerate(reversed(stages)):
        for p in s.parameters():
            p.requires_grad = (i < n)
    uf = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model.backbone.parameters())
    print(f"  Unfreeze {n}/{len(stages)} stages ({uf:,}/{tt:,} backbone params)")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class HybridDataset(Dataset):
    def __init__(self, paths, labels, transform=None, cache=None):
        self.paths   = paths
        self.labels  = labels
        self.transform = transform
        self.cache   = cache or {}

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p   = self.paths[idx]
        img = Image.open(p).convert('RGB')
        if self.transform: img = self.transform(img)

        med_feat = self.cache.get(p, np.zeros(NUM_FEATURES - 1, dtype=np.float32))
        is_syn = 1.0 if 'syn_' in Path(p).name else 0.0
        feat   = np.append(med_feat, is_syn).astype(np.float32)

        return {
            'image':    img,
            'features': torch.FloatTensor(feat),
            'label':    torch.tensor(self.labels[idx], dtype=torch.long),
            'path':     p,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def train_transform(sz=300):
    return transforms.Compose([
        transforms.RandomResizedCrop(sz, scale=(0.7,1.0), ratio=(0.85,1.15),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.06),
        transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.1), shear=8),
        transforms.RandomGrayscale(p=0.06),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02,0.12)),
    ])


def val_transform(sz=300):
    return transforms.Compose([
        transforms.Resize((sz,sz), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])


def tta_transforms(sz=300):
    n = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    r = [transforms.Resize((sz,sz),interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.ToTensor(), n]
    return [
        transforms.Compose(r),
        transforms.Compose([transforms.RandomHorizontalFlip(1.)]+r),
        transforms.Compose([transforms.RandomVerticalFlip(1.)]+r),
        transforms.Compose([transforms.RandomRotation((90,90))]+r),
        transforms.Compose([transforms.RandomRotation((180,180))]+r),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MixUp — FIXED: unsqueeze(1) on mask for feature broadcasting
# ─────────────────────────────────────────────────────────────────────────────
def adjacent_mixup(images, features, labels, alpha=0.1):
    """Mix only samples with |label_a - label_b| <= 1 AND same domain."""
    B   = images.size(0)
    lam = float(np.random.beta(alpha, alpha))
    lam = max(0.1, min(0.9, lam))
    perm = torch.randperm(B, device=images.device)
    la, lb = labels, labels[perm]

    adj      = (torch.abs(la.float()-lb.float()) <= 1).float()
    same_dom = (features[:, -1] == features[perm, -1]).float()
    mask_1d  = adj * same_dom                          # (B,)

    img_mask  = mask_1d.view(-1, 1, 1, 1)             # (B,1,1,1)  for images
    feat_mask = mask_1d.view(-1, 1)                    # (B,1)      for features

    mixed_img  = img_mask  * (lam*images   + (1-lam)*images[perm])   + (1-img_mask) *images
    mixed_feat = feat_mask * (lam*features + (1-lam)*features[perm]) + (1-feat_mask)*features

    return mixed_img, mixed_feat, la, lb, lam, mask_1d

def cutmix_fn(images, features, labels, alpha=0.1):
    B, _, H, W = images.shape
    lam = float(np.random.beta(alpha, alpha))
    lam = max(0.1, min(0.9, lam))
    idx = torch.randperm(B, device=images.device)
    r   = math.sqrt(1-lam)
    ch, cw = int(H*r), int(W*r)
    cx, cy = random.randint(0,W), random.randint(0,H)
    x1,x2 = max(0,cx-cw//2), min(W,cx+cw//2)
    y1,y2 = max(0,cy-ch//2), min(H,cy+ch//2)
    mixed = images.clone()
    mixed[:,:,y1:y2,x1:x2] = images[idx,:,y1:y2,x1:x2]
    lam = 1-(x2-x1)*(y2-y1)/(H*W)
    return mixed, lam*features+(1-lam)*features[idx], labels, labels[idx], lam


def get_aug_params(epoch):
    if epoch < 40:
        return False, False, 0.
    alpha = min(0.05, 0.02 + 0.002*(epoch-40))
    return True, epoch >= 50, alpha


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────
class WarmCosine:
    def __init__(self, opt, warmup, total, min_frac=0.05):
        self.opt=opt; self.warmup=warmup; self.total=total
        self.min_f=min_frac; self.ep=0
        for pg in opt.param_groups: pg['base_lr']=pg['lr']

    def step(self):
        self.ep += 1
        e = self.ep
        if e <= self.warmup:
            s = e/max(1,self.warmup)
        else:
            prog = (e-self.warmup)/max(1,self.total-self.warmup)
            s = self.min_f + 0.5*(1-self.min_f)*(1+math.cos(math.pi*prog))
        for pg in self.opt.param_groups:
            pg['lr'] = pg['base_lr']*s

    def lrs(self): return [pg['lr'] for pg in self.opt.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
# SWA BN update helpers
# ─────────────────────────────────────────────────────────────────────────────
def _reset_bn(swa_model):
    for m in swa_model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()


@torch.no_grad()
def mini_bn_update(loader, swa_model, device, n=60):
    _reset_bn(swa_model)
    swa_model.train()
    for i, b in enumerate(loader):
        if i >= n: break
        swa_model(b['image'].to(device), b['features'].to(device))
    swa_model.eval()


@torch.no_grad()
def full_bn_update(loader, swa_model, device):
    _reset_bn(swa_model)
    swa_model.train()
    for b in tqdm(loader, desc="SWA BN", leave=False):
        swa_model(b['image'].to(device), b['features'].to(device))
    swa_model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, ce_fn, ord_fn, device, scaler, epoch):
    model.train()
    tot_loss=cor=tot=0
    first=True
    use_mix, use_cut, alpha = get_aug_params(epoch)

    for batch in tqdm(loader, desc="Training", leave=False):
        imgs  = batch['image'].to(device, non_blocking=True)
        feats = batch['features'].to(device, non_blocking=True)
        lbls  = batch['label'].to(device, non_blocking=True)

        if first:
            first=False
            fmin,fmax = feats[:,:-1].min().item(), feats[:,:-1].max().item()
            syn_pct   = feats[:,-1].mean().item()*100
            ok = "✅" if not (abs(fmin)<1e-6 and abs(fmax)<1e-6) else "❌ ALL ZERO"
            print(f"  feat[0:30] [{fmin:.2f},{fmax:.2f}] {ok} | syn_flag={syn_pct:.0f}%")

        lam=1.; la=lb=lbls; adj_mask=None

        if use_mix or use_cut:
            if use_cut and USE_CUTMIX and random.random()>0.5:
                imgs,feats,la,lb,lam = cutmix_fn(imgs,feats,lbls,alpha)
            elif use_mix:
                imgs,feats,la,lb,lam,adj_mask = adjacent_mixup(imgs,feats,lbls,alpha)

        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits, ord_logits = model(imgs, feats)

            if lam < 1.:
                if adj_mask is not None:
                    ce_a = ce_fn(logits, la)
                    ce_b = ce_fn(logits, lb)
                    ce   = (lam*ce_a + (1-lam)*ce_b)
                    ol   = ord_fn(ord_logits, la)
                else:
                    ce = lam*ce_fn(logits,la) + (1-lam)*ce_fn(logits,lb)
                    ol = lam*ord_fn(ord_logits,la) + (1-lam)*ord_fn(ord_logits,lb)
            else:
                ce = ce_fn(logits, lbls)
                ol = ord_fn(ord_logits, lbls)

            loss = CE_WEIGHT*ce + ORDINAL_WEIGHT*ol

        if not torch.isfinite(loss):
            print("  ⚠️  non-finite loss, skip")
            continue

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

        tot_loss += loss.item()
        tot      += lbls.size(0)
        cor      += (logits.detach().argmax(1)==lbls).sum().item()

    return tot_loss/max(1,len(loader)), 100.*cor/max(1,tot)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, ce_fn, class_names,
             use_tta=False, tta_t=None, verbose=False):
    model.eval()
    preds,tgts,tot_loss = [],[],0.

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            imgs  = batch['image'].to(device, non_blocking=True)
            feats = batch['features'].to(device, non_blocking=True)
            lbls  = batch['label'].to(device, non_blocking=True)

            if use_tta and tta_t:
                ls = None
                for t in tta_t:
                    aug = torch.stack([t(Image.open(p).convert('RGB'))
                                       for p in batch['path']]).to(device)
                    out = model(aug, feats, return_ordinal=False)
                    ls  = out if ls is None else ls+out
                logits = ls/len(tta_t)
            else:
                logits = model(imgs, feats, return_ordinal=False)

            l = ce_fn(logits, lbls)
            if torch.isfinite(l): tot_loss += l.item()
            preds.extend(logits.argmax(1).cpu().numpy())
            tgts.extend(lbls.cpu().numpy())

    acc  = 100.*sum(p==t for p,t in zip(preds,tgts))/max(1,len(tgts))
    bacc = 100.*balanced_accuracy_score(tgts, preds)
    loss = tot_loss/max(1,len(loader))
    rep  = classification_report(tgts, preds, target_names=class_names,
                                  output_dict=True, zero_division=0)
    f1   = rep['macro avg']['f1-score']

    if verbose:
        print("\n  Per-class results:")
        for i,n in enumerate(class_names):
            ct = [t for t in tgts if t==i]
            cp = [p for p,t in zip(preds,tgts) if t==i]
            ca = 100*sum(p==i for p in cp)/max(1,len(ct))
            pr = rep.get(n,{})
            print(f"    {n:10s}: acc={ca:5.1f}%  F1={pr.get('f1-score',0):.3f}"
                  f"  P={pr.get('precision',0):.3f}  R={pr.get('recall',0):.3f} (n={len(ct)})")
        cm = confusion_matrix(tgts, preds)
        print("\n  Confusion matrix (rows=true, cols=pred):")
        hdr = "            " + "".join(f"{c[:8]:>9}" for c in class_names)
        print(hdr)
        for i,row in enumerate(cm):
            print(f"  {class_names[i][:9]:9s}  "+"".join(f"{v:9d}" for v in row))

    return acc, bacc, loss, f1, preds, tgts


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def sanitize(arr):
    return np.clip(np.nan_to_num(np.array(arr,np.float32),
                                 nan=0.,posinf=0.,neginf=0.), -1e6, 1e6)


def build_cache(paths, scaler=None, fit=False):
    raw,nbad = {},0
    N_MED = NUM_FEATURES - 1
    for p in tqdm(paths, desc="Features", leave=False):
        try:
            f = extract_medical_features(Image.open(p).convert('RGB'))
            f = sanitize(f)
            if len(f) != N_MED:
                raise ValueError(f"expected {N_MED} features, got {len(f)}")
        except Exception:
            f = np.zeros(N_MED, np.float32); nbad+=1
        raw[p]=f

    if nbad:
        pct=100*nbad/max(1,len(paths))
        print(f"  ⚠️  {nbad} ({pct:.1f}%) feature failures")
        if pct>30: print("  ❌  >30% failures — check feature_extractor!")

    sample = np.concatenate([raw[p] for p in list(raw)[:10]])
    if np.allclose(sample,0.):
        print("  ❌  CRITICAL: all features zero!")
    else:
        print(f"  ✅  Feature range [{sample.min():.3f},{sample.max():.3f}]")

    if fit:
        scaler = RobustScaler(quantile_range=(10.,90.))
        scaler.fit(np.stack([raw[p] for p in paths]))

    cache = {}
    for p in paths:
        if scaler is not None:
            cache[p] = sanitize(np.clip(scaler.transform([raw[p]])[0], -3.,3.))
        else:
            cache[p] = raw[p]
    return cache, scaler


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def train(data_dir, output_dir, epochs=100, batch_size=32,
          lr=2e-4, patience=20, num_workers=4, resume=False):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    dp = Path(data_dir)

    # ── Load data ─────────────────────────────────────────────────────────
    trp, vap = dp/'train', dp/'val'

    def load_split(sp, cls_list):
        paths,lbls=[],[]
        for i,c in enumerate(cls_list):
            d=sp/c
            if not d.exists(): continue
            imgs=(sorted(d.glob('*.jpg'))+sorted(d.glob('*.JPG'))+
                  sorted(d.glob('*.png'))+sorted(d.glob('*.PNG'))+
                  sorted(d.glob('*.jpeg'))+sorted(d.glob('*.bmp'))+
                  sorted(d.glob('*.BMP')))
            print(f"     {c}: {len(imgs)}")
            for x in imgs: paths.append(str(x)); lbls.append(i)
        return paths,lbls

    if trp.exists() and vap.exists():
        cls = sorted([p.name for p in trp.iterdir() if p.is_dir()])
        print(f"✅ Classes found (alphabetical): {cls}")
        print("📂 train/"); tr_p,tr_l = load_split(trp,cls)
        print("📂 val/");   va_p,va_l = load_split(vap,cls)
        vc = Counter(va_l)
        if len(vc) > 0 and max(vc.values())/max(min(vc.values()),1) > 3:
            print(f"⚠️  Val imbalanced — re-splitting 80/20 stratified")
            ap,al = tr_p+va_p, tr_l+va_l
            tr_p,va_p,tr_l,va_l = train_test_split(
                ap,al,test_size=0.2,random_state=42,stratify=al)
            tc,vc2 = Counter(tr_l),Counter(va_l)
            for i,n in enumerate(cls):
                print(f"   {n}: train={tc[i]}  val={vc2[i]}")
    else:
        cls = sorted([p.name for p in dp.iterdir()
                      if p.is_dir() and p.name not in
                      ('test','synthetic','sipakmed_raw','__pycache__')])
        print(f"✅ Classes found (flat): {cls}")
        ap,al = load_split(dp,cls)
        tr_p,va_p,tr_l,va_l = train_test_split(ap,al,test_size=0.2,
                                                 random_state=42,stratify=al)

    print(f"\n✅ Train: {len(tr_p)}  Val: {len(va_p)}")
    if not tr_p: raise ValueError("No training images found.")

    sev_present = [c for c in SEVERITY_ORDER if c in cls]
    if set(sev_present) == set(cls):
        old2new = {cls.index(c): sev_present.index(c) for c in cls}
        tr_l = [old2new[l] for l in tr_l]
        va_l = [old2new[l] for l in va_l]
        cls  = sev_present
        print(f"✅ Labels remapped to severity order: {cls}")
    else:
        missing = set(cls) - set(SEVERITY_ORDER)
        print(f"⚠️  Unknown classes {missing} — ordinal loss may be less effective.")
        print(f"   Expected: {SEVERITY_ORDER}")

    syn_tr = sum(1 for p in tr_p if 'syn_' in Path(p).name)
    syn_va = sum(1 for p in va_p if 'syn_' in Path(p).name)
    print(f"   Synthetic images — train: {syn_tr} ({100*syn_tr/max(1,len(tr_p)):.1f}%)"
          f"  val: {syn_va} ({100*syn_va/max(1,len(va_p)):.1f}%)")

    # ── Features ──────────────────────────────────────────────────────────
    print("\n🔍 Extracting train features (30 medical dims)...")
    tr_cache, scaler = build_cache(tr_p, fit=True)
    print("🔍 Extracting val features...")
    va_cache, _      = build_cache(va_p, scaler=scaler)

    # ── Datasets & loaders ────────────────────────────────────────────────
    tr_ds = HybridDataset(tr_p, tr_l, train_transform(INPUT_SIZE), tr_cache)
    va_ds = HybridDataset(va_p, va_l, val_transform(INPUT_SIZE),   va_cache)
    tta_t = tta_transforms(INPUT_SIZE) if USE_TTA else None

    tc       = Counter(tr_l)
    hard_idx = {cls.index(c) for c in HARD_CLASSES if c in cls}
    sw       = [(1./tc[l])*(3. if l in hard_idx else 1.) for l in tr_l]
    sampler  = WeightedRandomSampler(sw, len(tr_l), replacement=True)

    nw = min(num_workers, os.cpu_count() or 0)
    tr_loader = DataLoader(tr_ds, batch_size, sampler=sampler, num_workers=nw,
                           pin_memory=True, drop_last=True, persistent_workers=(nw>0))
    va_loader = DataLoader(va_ds, batch_size, shuffle=False, num_workers=nw,
                           pin_memory=True, persistent_workers=(nw>0))

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(len(cls), NUM_FEATURES, device, dropout=0.3, dpr=0.2)
    freeze_backbone(model)

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    best_f1 = best_bacc = best_acc = 0.
    if resume:
        ckpt_path = os.path.join(output_dir, 'best_model.pt')
        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck['model_state_dict'])
            start_epoch = ck.get('epoch', 0)
            best_f1     = ck.get('macro_f1', 0.)
            best_bacc   = ck.get('val_bacc', 0.)
            best_acc    = ck.get('val_acc',  0.)
            print(f"▶️  Resumed from epoch {start_epoch}  F1={best_f1:.4f}  bal={best_bacc:.2f}%")
        else:
            print("⚠️  No checkpoint found — starting from scratch")

    # ── Optimizer helpers ─────────────────────────────────────────────────
    bb_ids = {id(p) for p in model.backbone.parameters()}
    head_p = [p for p in model.parameters() if id(p) not in bb_ids]
    back_p = list(model.backbone.parameters())

    def make_opt(bb_lr, hd_lr):
        return torch.optim.AdamW([
            {'params': back_p, 'lr': bb_lr, 'weight_decay': 1e-4},
            {'params': head_p, 'lr': hd_lr, 'weight_decay': 1e-4},
        ])

    # Create optimizer BEFORE the training loop
    opt = make_opt(lr * 0.01, lr)
    sch = WarmCosine(opt, warmup=2, total=epochs-start_epoch, min_frac=0.03)

    # ── Loss ──────────────────────────────────────────────────────────────
    ce_fn  = FocalLoss(gamma=1.0, smoothing=0.08, num_classes=len(cls))
    ord_fn = OrdinalLoss(num_classes=len(cls), smoothing=0.05)

    # ── AMP / SWA ─────────────────────────────────────────────────────────
    amp_scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    swa_model  = AveragedModel(model) if USE_SWA else None

    print(f"\n{'='*70}")
    print(f"  v11 — SIPaKMeD 4-class config:")
    print(f"  Classes       : {cls}")
    print(f"  Loss          : {CE_WEIGHT}xFocalLoss + {ORDINAL_WEIGHT}xOrdinalLoss")
    print(f"  Features      : {NUM_FEATURES} (30 medical + 1 synthetic flag)")
    print(f"  Hard classes  : {HARD_CLASSES} → 3x sampler weight")
    print(f"  MixUp         : adjacent-grade only, same-domain, starts ep20")
    print(f"  SWA start     : epoch {SWA_START}")
    print(f"  Backbone      : 1 stage unfrozen per {UNFREEZE_STEP} epochs")
    print(f"{'='*70}\n")

    pat=0; unfrz=False
    ckpt = os.path.join(output_dir, 'best_model.pt')
    hist = {k:[] for k in ['tr_loss','tr_acc','va_loss','va_acc','va_f1','va_bacc']}

    for ep in range(start_epoch, epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {ep+1}/{epochs}")
        print(f"{'='*70}")

        # Progressive unfreeze
        if not unfrz and ep >= FREEZE_EPOCHS:
            unfrz = True
            unfreeze_progressive(model, ep, FREEZE_EPOCHS)
            
            # Rebuild optimizer with unfrozen backbone params
            bb_ids = {id(p) for p in model.backbone.parameters()}
            head_p = [p for p in model.parameters() if id(p) not in bb_ids]
            back_p = list(model.backbone.parameters())
            
            opt = make_opt(lr * 0.005, lr)
            for pg in opt.param_groups:
                pg['base_lr'] = pg['lr']
            
            sch = WarmCosine(opt, warmup=2, total=epochs-ep, min_frac=0.03)
            if USE_SWA:
                swa_model = AveragedModel(model)
            print(f"  Rebuilt optimizer (bb_lr={lr*0.005:.1e}, head_lr={lr:.1e})")
        elif unfrz and ep > FREEZE_EPOCHS:
            unfreeze_progressive(model, ep, FREEZE_EPOCHS)

        # Train
        tr_loss, tr_acc = train_epoch(
            model, tr_loader, opt, ce_fn, ord_fn, device, amp_scaler, ep)

        # SWA
        use_swa = USE_SWA and ep >= SWA_START
        if use_swa:
            swa_model.update_parameters(model)
            mini_bn_update(tr_loader, swa_model, device, n=60)

        sch.step()

        # Eval
        do_tta  = USE_TTA and (ep+1)%TTA_EVERY==0
        verbose = True
        eval_m  = swa_model if use_swa else model

        va_acc,va_bacc,va_loss,f1,_,_ = evaluate(
            eval_m, va_loader, device, ce_fn, cls,
            use_tta=do_tta, tta_t=tta_t, verbose=verbose)

        lrs = sch.lrs()
        _,_,alpha = get_aug_params(ep)
        tags = ("" + (" [TTA]" if do_tta else "") + (" [SWA]" if use_swa else ""))
        print(f"Train  — loss:{tr_loss:.4f} | acc:{tr_acc:.2f}%")
        print(f"Val    — loss:{va_loss:.4f} | acc:{va_acc:.2f}%"
              f" | bal:{va_bacc:.2f}% | F1:{f1:.4f}{tags}")
        print(f"LR     — head={lrs[1]:.2e} bb={lrs[0]:.2e} | alpha={alpha:.3f}")

        for k,v in zip(['tr_loss','tr_acc','va_loss','va_acc','va_f1','va_bacc'],
                       [tr_loss,tr_acc,va_loss,va_acc,f1,va_bacc]):
            hist[k].append(v)

        # Checkpoint
        improved = (f1 > best_f1+5e-4) or \
                   (va_bacc > best_bacc+0.3 and f1 > best_f1-0.01) or \
                   (va_acc  > best_acc +0.5 and f1 > best_f1-0.005)
        if improved:
            best_f1=f1; best_bacc=va_bacc; best_acc=va_acc; pat=0
            sm = swa_model if use_swa else model
            torch.save({
                'epoch':            ep+1,
                'model_state_dict': sm.module.state_dict() if use_swa else sm.state_dict(),
                'val_acc':          va_acc,
                'val_bacc':         va_bacc,
                'macro_f1':         f1,
                'class_names':      cls,
                'num_classes':      len(cls),
                'num_features':     NUM_FEATURES,
                'feat_dim':         FEAT_DIM,
                'input_size':       INPUT_SIZE,
                'version':          _VERSION,
                'use_swa':          use_swa,
                'severity_order':   SEVERITY_ORDER,
            }, ckpt)
            print(f"✅ Saved (F1={f1:.4f}, acc={va_acc:.2f}%, bal={va_bacc:.2f}%)")
        else:
            pat+=1
            print(f"   No improvement ({pat}/{patience})")
            if pat>=patience:
                print(f"\n⏹️  Early stopping at ep{ep+1}")
                break

    # Final SWA BN update
    if USE_SWA and swa_model is not None and ep>=SWA_START:
        print("\n🔄 Final SWA BN update...")
        full_bn_update(tr_loader, swa_model, device)
        sa,sb,sl,sf,_,_ = evaluate(swa_model,va_loader,device,ce_fn,cls,
                                    use_tta=USE_TTA,tta_t=tta_t,verbose=True)
        print(f"  SWA final — acc={sa:.2f}% bal={sb:.2f}% F1={sf:.4f}")
        sp=os.path.join(output_dir,'swa_model.pt')
        torch.save({
            'model_state_dict': swa_model.module.state_dict(),
            'class_names':      cls,
            'num_classes':      len(cls),
            'num_features':     NUM_FEATURES,
            'feat_dim':         FEAT_DIM,
            'input_size':       INPUT_SIZE,
            'version':          _VERSION,
            'severity_order':   SEVERITY_ORDER,
        }, sp)
        print(f"  SWA saved: {sp}")
        if sf>best_f1 or sb>best_bacc:
            import shutil; shutil.copy(sp,ckpt)
            print(f"  ✅ SWA is best — copied to {ckpt}")

    with open(os.path.join(output_dir,'history.json'),'w') as f:
        json.dump(hist,f,indent=2)

    print(f"\n{'='*70}")
    print(f"✅ Done! best_acc={best_acc:.2f}% balanced={best_bacc:.2f}% F1={best_f1:.4f}")
    print(f"{'='*70}")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',                type=str,   default='/kaggle/working/data_final')
    parser.add_argument('--checkpoint-dir',          type=str,   default='./checkpoints')
    parser.add_argument('--epochs',                  type=int,   default=100)
    parser.add_argument('--batch-size',              type=int,   default=32)
    parser.add_argument('--learning-rate',           type=float, default=1e-4)
    parser.add_argument('--early-stopping-patience', type=int,   default=20)
    parser.add_argument('--num-workers',             type=int,   default=4)
    parser.add_argument('--seed',                    type=int,   default=42)
    parser.add_argument('--no-tta',    action='store_true')
    parser.add_argument('--no-swa',    action='store_true')
    parser.add_argument('--use-sam',   action='store_true')
    parser.add_argument('--no-cutmix', action='store_true')
    parser.add_argument('--resume',    action='store_true',
                        help='Resume training from best_model.pt in checkpoint-dir')
    args = parser.parse_args()

    if args.no_tta:    USE_TTA    = False
    if args.no_swa:    USE_SWA    = False
    if args.use_sam:   USE_SAM    = True
    if args.no_cutmix: USE_CUTMIX = False

    set_seed(args.seed)

    print(f"\n{'='*70}")
    print(f"v11 — SIPaKMeD + Herlev | 4-class ordinal")
    print(f"{'='*70}")
    for k,v in vars(args).items():
        print(f"  {k:<30}: {v}")
    print(f"{'='*70}\n")

    train(
        data_dir    = args.data_dir,
        output_dir  = args.checkpoint_dir,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.learning_rate,
        patience    = args.early_stopping_patience,
        num_workers = args.num_workers,
        resume      = args.resume,
    )