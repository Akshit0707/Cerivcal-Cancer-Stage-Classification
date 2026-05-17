"""
load_model.py — loader for EfficientNetHybrid v11 checkpoints.

Infers ALL architecture dimensions from the checkpoint's own weight shapes
so it works regardless of which training-script version produced the file.

Compatible state-dict layouts
──────────────────────────────
  v11  (train_hybrid.py)  : feat_mlp.*, se.*, head.*, ordinal_head.*
  legacy (older scripts)  : feature_mlp.*, attention.*, classifier.*
  SWA wrapper             : module.<key> prefix — stripped automatically
"""

import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── Shared constants ──────────────────────────────────────────────────────────
NUM_FEATURES   = 31
SEVERITY_ORDER = ['Normal', 'CIN1', 'HighGrade', 'Cancer']


# ── Legacy pickle aliases (must run before torch.load) ────────────────────────

def _register_legacy_aliases():
    try:
        _here = Path(__file__).resolve().parent
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))

        import model as _mod

        for alias in [
            'hybrid_model',
            'hybrid_model.CPUOptimizedHybridModel',
            'backend.hybrid_model',
            'backend.hybrid_model.CPUOptimizedHybridModel',
            'cpu_model',
            'cpu_model.CPUOptimizedHybridModel',
        ]:
            sys.modules.setdefault(alias, _mod)

        sys.modules['hybrid_model.EfficientNetHybrid'] = _mod.EfficientNetHybrid
        sys.modules['hybrid_model.FeatureMLP']         = _mod.FeatureMLP

    except ImportError as e:
        print(f'[load_model] WARNING: could not register legacy aliases: {e}')


_register_legacy_aliases()


# ── Dimension inference ───────────────────────────────────────────────────────

def _infer_dims(sd: dict) -> dict:
    """
    Read tensor shapes to recover every architecture hyperparameter.

    Handles both v11 key names (feat_mlp / se / head / ordinal_head)
    and legacy key names (feature_mlp / attention / classifier).

    Returns dict with keys:
      num_classes, num_features, feat_dim,
      has_ordinal_head, is_v11
    """
    dims = {}

    # ── Detect layout version ────────────────────────────────────────────
    is_v11 = any(k.startswith('feat_mlp.') for k in sd) or \
              any(k.startswith('se.')       for k in sd) or \
              any(k.startswith('head.')     for k in sd)
    dims['is_v11'] = is_v11

    # ── num_features — input dim of first FeatureMLP linear ─────────────
    for k in [
        'feat_mlp.net.0.weight',          # v11
        'feature_mlp.net.0.weight',       # legacy
        'module.feat_mlp.net.0.weight',
        'module.feature_mlp.net.0.weight',
    ]:
        if k in sd:
            dims['num_features'] = sd[k].shape[1]
            break
    dims.setdefault('num_features', NUM_FEATURES)

    # ── feat_dim — FeatureMLP output dim ────────────────────────────────
    # v11: feat_mlp.net.8.weight  shape [feat_dim, 256]
    # legacy: feature_mlp.net.8.weight or proj.weight
    for k in [
        'feat_mlp.net.8.weight',
        'feature_mlp.net.8.weight',
        'feature_mlp.proj.weight',
        'module.feat_mlp.net.8.weight',
        'module.feature_mlp.net.8.weight',
        'module.feature_mlp.proj.weight',
    ]:
        if k in sd:
            dims['feat_dim'] = sd[k].shape[0]
            break
    dims.setdefault('feat_dim', 128)

    # ── num_classes ──────────────────────────────────────────────────────
    # v11: head.9.weight  shape [num_classes, 256]
    # legacy: classifier.5.weight or classifier.4.weight
    for k in [
        'head.9.weight',
        'head.5.weight',                  # 2-linear head fallback
        'classifier.5.weight',
        'classifier.4.weight',
        'module.head.9.weight',
        'module.head.5.weight',
        'module.classifier.5.weight',
        'module.classifier.4.weight',
    ]:
        if k in sd:
            dims['num_classes'] = sd[k].shape[0]
            break
    dims.setdefault('num_classes', 4)

    # ── ordinal_head present? ────────────────────────────────────────────
    dims['has_ordinal_head'] = any(
        k.startswith('ordinal_head.') or k.startswith('module.ordinal_head.')
        for k in sd
    )

    return dims


# ── Model builder that handles both layouts ───────────────────────────────────

def _build_v11(sd: dict, dims: dict, dropout: float, dpr: float):
    """Build the v11 EfficientNetHybrid architecture."""
    from model import EfficientNetHybrid
    return EfficientNetHybrid(
        num_classes    = dims['num_classes'],
        num_features   = dims['num_features'],
        feat_dim       = dims['feat_dim'],
        dropout        = dropout,
        drop_path_rate = dpr,
    )


def _build_legacy(sd: dict, dims: dict, dropout: float, dpr: float):
    """
    Build a legacy EfficientNetHybrid (attention-gate, no ordinal head)
    using inline class definitions so we don't need a separate file.
    Only used when loading pre-v11 checkpoints.
    """
    import torch.nn as nn
    from torchvision import models as tvm

    class _FeatureMLP(nn.Module):
        def __init__(self, in_dim, out_dim, h1=256, h2=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(h1, h2),    nn.BatchNorm1d(h2), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(h2, out_dim), nn.BatchNorm1d(out_dim),
            )
            self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        def forward(self, x): return self.net(x) + self.proj(x)

    class _Legacy(nn.Module):
        def __init__(self, nc, nf, fd, dropout, dpr):
            super().__init__()
            bb = tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.cnn_dim = bb.classifier[1].in_features
            bb.classifier = nn.Identity()
            self.backbone = bb
            self.feature_mlp = _FeatureMLP(nf, fd)
            fusion = self.cnn_dim + fd
            self.attention = nn.Sequential(
                nn.Linear(fusion, 64), nn.GELU(),
                nn.Linear(64, 2),     nn.Softmax(dim=-1),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(fusion, 512), nn.BatchNorm1d(512), nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(512, nc),
            )
        def forward(self, images, features, return_ordinal=False):
            images   = torch.nan_to_num(images,   nan=0., posinf=1.,  neginf=-1.)
            features = torch.nan_to_num(features, nan=0., posinf=0.,  neginf=0.)
            cnn  = self.backbone(images)
            feat = self.feature_mlp(features)
            fused = torch.cat([cnn, feat], dim=1)
            attn  = self.attention(fused)
            fused_scaled = torch.cat(
                [cnn * attn[:, 0:1], feat * attn[:, 1:2]], dim=1)
            logits = self.classifier(fused_scaled)
            if return_ordinal:
                return logits, attn   # attn used as pseudo ord_logits for compat
            return logits

    return _Legacy(
        dims['num_classes'], dims['num_features'], dims['feat_dim'],
        dropout, dpr,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path:          str,
    num_classes:              int   = 4,
    num_traditional_features: int   = NUM_FEATURES,
    device:                   str   = 'cpu',
    dropout:                  float = 0.5,
    drop_path_rate:           float = 0.4,
):
    """
    Load EfficientNetHybrid from a v11 (or legacy) checkpoint.

    All architecture dims are inferred from the checkpoint's weight shapes —
    no hard-coding required.

    Args:
        checkpoint_path          : Path to .pt / .pth file
        num_classes              : Fallback if absent from checkpoint (default 4)
        num_traditional_features : Fallback if absent from checkpoint (default 31)
        device                   : 'cpu' or 'cuda'
        dropout                  : Reconstruction dropout (default matches training)
        drop_path_rate           : Reconstruction DPR     (default matches training)

    Returns:
        EfficientNetHybrid in eval() mode
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'[load_model] Checkpoint not found: {checkpoint_path}\n'
            f'  cwd: {os.getcwd()}'
        )

    print(f'[load_model] Loading: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # ── Unwrap checkpoint dict vs raw state-dict ──────────────────────────
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd          = ckpt['model_state_dict']
        class_names = ckpt.get('class_names', [])
        ckpt_cls    = len(class_names) if class_names else ckpt.get('num_classes')
        ckpt_feats  = ckpt.get('num_features')
        val_acc     = ckpt.get('val_acc')
        macro_f1    = ckpt.get('macro_f1')
        epoch       = ckpt.get('epoch')
        is_swa      = ckpt.get('use_swa', False)
    else:
        print('[load_model] Raw state-dict checkpoint (no metadata wrapper)')
        sd = ckpt
        class_names = []
        ckpt_cls = ckpt_feats = val_acc = macro_f1 = epoch = None
        is_swa = False

    # ── Strip SWA 'module.' prefix ────────────────────────────────────────
    if any(k.startswith('module.') for k in sd):
        print("[load_model] SWA checkpoint — stripping 'module.' prefix")
        sd     = {k.replace('module.', '', 1): v for k, v in sd.items()}
        is_swa = True

    # ── Infer dims ────────────────────────────────────────────────────────
    inferred = _infer_dims(sd)
    print(f'[load_model] Inferred dims : {inferred}')

    # Checkpoint metadata overrides inference when available
    if ckpt_cls:
        inferred['num_classes']  = ckpt_cls
    if ckpt_feats:
        inferred['num_features'] = ckpt_feats
    # CLI fallbacks (only when nothing else resolved them)
    inferred.setdefault('num_classes',  num_classes)
    inferred.setdefault('num_features', num_traditional_features)

    # ── Build matching architecture ───────────────────────────────────────
    if inferred['is_v11']:
        model = _build_v11(sd, inferred, dropout, drop_path_rate)
        print('[load_model] Layout: v11  (feat_mlp / se / head / ordinal_head)')
    else:
        model = _build_legacy(sd, inferred, dropout, drop_path_rate)
        print('[load_model] Layout: legacy  (feature_mlp / attention / classifier)')

    # ── Load weights ──────────────────────────────────────────────────────
    try:
        model.load_state_dict(sd, strict=True)
        print('[load_model] Weights loaded (strict=True)')
    except RuntimeError:
        mk = set(model.state_dict())
        ck = set(sd)
        missing    = sorted(mk - ck)
        unexpected = sorted(ck - mk)
        if missing:
            print(f'[load_model] Missing keys    ({len(missing)}): '
                  f'{missing[:8]}{"..." if len(missing) > 8 else ""}')
        if unexpected:
            print(f'[load_model] Unexpected keys ({len(unexpected)}): '
                  f'{unexpected[:8]}{"..." if len(unexpected) > 8 else ""}')
        model.load_state_dict(sd, strict=False)
        print('[load_model] Weights loaded (strict=False — some layers randomised)')

    model.to(device).eval()

    print(f'[load_model] Model ready')
    print(f'  Classes      : {class_names if class_names else inferred["num_classes"]}')
    print(f'  num_features : {inferred["num_features"]}')
    print(f'  feat_dim     : {inferred["feat_dim"]}')
    print(f'  fusion_dim   : {1536 + inferred["feat_dim"]}')
    print(f'  ordinal_head : {inferred["has_ordinal_head"]}')
    if val_acc  is not None: print(f'  Val Acc      : {val_acc:.2f}%')
    if macro_f1 is not None: print(f'  Macro-F1     : {macro_f1:.4f}')
    if epoch    is not None: print(f'  Epoch        : {epoch}')
    if is_swa:               print(f'  SWA          : Yes')

    return model


def get_class_names() -> list:
    """4-class ordinal label names used by train_hybrid.py v11."""
    return list(SEVERITY_ORDER)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', nargs='?',
                   default='checkpoints/best_model.pt')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    model = load_model(args.checkpoint, device=args.device)

    with torch.no_grad():
        # return_ordinal=True  → (logits, ord_logits)
        out = model(
            torch.randn(1, 3, 300, 300).to(args.device),
            torch.randn(1, NUM_FEATURES).to(args.device),
            return_ordinal=True,
        )

    if isinstance(out, tuple):
        logits, ord_logits = out
        print(f'\nSmoke test — logits {logits.shape}  ord_logits {ord_logits.shape}')
    else:
        logits = out
        print(f'\nSmoke test — logits {logits.shape}')

    probs = F.softmax(logits, dim=-1)[0]
    for name, prob in zip(get_class_names(), probs.tolist()):
        print(f'  {name:<12} {prob:.4f}')

    print('\n✅ Smoke test passed.')