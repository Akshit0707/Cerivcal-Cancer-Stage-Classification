"""
Model loader for EfficientNetHybrid checkpoints saved by train_hybrid.py.

Infers ALL architecture dimensions from the checkpoint's own weight shapes
so it works regardless of which training script version produced the file.
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
from pathlib import Path


# ---------------------------------------------------------------------------
# Register legacy pickle aliases BEFORE any torch.load call
# ---------------------------------------------------------------------------
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
        print(f"[load_model] WARNING: could not register legacy aliases: {e}")


_register_legacy_aliases()


# ---------------------------------------------------------------------------
# Dimension inference from state-dict weight shapes
# ---------------------------------------------------------------------------
def _infer_dims(sd: dict) -> dict:
    """
    Read tensor shapes to recover every architecture hyperparameter.
    Returns a dict with keys:
      num_classes, num_features,
      mlp_hidden1, mlp_hidden2, feat_out_dim,
      attn_hidden, attn_dropout
    """
    dims = {}

    # num_classes -- last linear in classifier
    for k in ['classifier.5.weight', 'classifier.4.weight',
              'module.classifier.5.weight', 'module.classifier.4.weight']:
        if k in sd:
            dims['num_classes'] = sd[k].shape[0]
            break

    # num_features -- input to first FeatureMLP linear
    for k in ['feature_mlp.net.0.weight', 'module.feature_mlp.net.0.weight']:
        if k in sd:
            dims['num_features'] = sd[k].shape[1]
            break

    # mlp_hidden1 -- output of net.0
    for k in ['feature_mlp.net.0.weight', 'module.feature_mlp.net.0.weight']:
        if k in sd:
            dims['mlp_hidden1'] = sd[k].shape[0]
            break

    # mlp_hidden2 -- output of net.4
    for k in ['feature_mlp.net.4.weight', 'module.feature_mlp.net.4.weight']:
        if k in sd:
            dims['mlp_hidden2'] = sd[k].shape[0]
            break

    # feat_out_dim -- output of proj (= output of net.8 = input to fusion)
    for k in ['feature_mlp.proj.weight', 'module.feature_mlp.proj.weight']:
        if k in sd:
            dims['feat_out_dim'] = sd[k].shape[0]
            break
    # fallback: net.8 output (if proj key absent)
    if 'feat_out_dim' not in dims:
        for k in ['feature_mlp.net.8.weight', 'module.feature_mlp.net.8.weight']:
            if k in sd:
                dims['feat_out_dim'] = sd[k].shape[0]
                break
    # fallback: net.4 output for 2-block checkpoints
    if 'feat_out_dim' not in dims:
        for k in ['feature_mlp.net.4.weight', 'module.feature_mlp.net.4.weight']:
            if k in sd:
                dims['feat_out_dim'] = sd[k].shape[0]
                break

    # attn_hidden -- output of attention.0
    for k in ['attention.0.weight', 'module.attention.0.weight']:
        if k in sd:
            dims['attn_hidden'] = sd[k].shape[0]
            break

    # attn_dropout -- True when attention has 5 submodules (Dropout at idx 2)
    # Without dropout: .0 Linear  .1 GELU  .2 Linear  .3 Softmax
    # With dropout:    .0 Linear  .1 GELU  .2 Dropout  .3 Linear  .4 Softmax
    dims['attn_dropout'] = (
        'attention.3.weight' in sd or 'module.attention.3.weight' in sd
    )

    return dims


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_model(
    checkpoint_path:          str,
    num_classes:              int = 5,
    num_traditional_features: int = 30,
    device:                   str = 'cpu',
):
    """
    Load EfficientNetHybrid from a checkpoint.

    All architecture dims are inferred from the checkpoint's own weight
    shapes -- no hard-coding required.

    Args:
        checkpoint_path          : Path to .pt / .pth file
        num_classes              : Fallback if absent from checkpoint
        num_traditional_features : Fallback if absent from checkpoint
        device                   : 'cpu' or 'cuda'

    Returns:
        EfficientNetHybrid in eval() mode
    """
    from model import EfficientNetHybrid, FeatureMLP

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"[load_model] Checkpoint not found: {checkpoint_path}\n"
            f"  cwd: {os.getcwd()}"
        )

    print(f"[load_model] Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # -- Unwrap checkpoint dict vs raw state-dict ---------------------------
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
        print("[load_model] Raw state-dict checkpoint (no metadata wrapper)")
        sd = ckpt
        class_names = []
        ckpt_cls = ckpt_feats = val_acc = macro_f1 = epoch = None
        is_swa = False

    # -- Strip SWA 'module.' prefix -----------------------------------------
    if any(k.startswith('module.') for k in sd):
        print("[load_model] SWA checkpoint -- stripping 'module.' prefix")
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
        is_swa = True

    # -- Infer all dims from weight shapes ----------------------------------
    inferred = _infer_dims(sd)
    print(f"[load_model] Inferred dims: {inferred}")

    num_classes              = ckpt_cls   or inferred.get('num_classes',  num_classes)
    num_traditional_features = ckpt_feats or inferred.get('num_features', num_traditional_features)
    feat_out_dim  = inferred.get('feat_out_dim', 128)
    mlp_hidden1   = inferred.get('mlp_hidden1',  256)
    mlp_hidden2   = inferred.get('mlp_hidden2',  256)
    attn_hidden   = inferred.get('attn_hidden',   64)
    attn_dropout  = inferred.get('attn_dropout', False)

    # -- Build model with exactly the inferred dims -------------------------
    model = EfficientNetHybrid(
        num_classes    = num_classes,
        num_features   = num_traditional_features,
        feat_out_dim   = feat_out_dim,
        mlp_hidden1    = mlp_hidden1,
        mlp_hidden2    = mlp_hidden2,
        attn_hidden    = attn_hidden,
        attn_dropout   = attn_dropout,
        dropout        = 0.5,
        drop_path_rate = 0.2,
    )

    # -- Load weights -------------------------------------------------------
    try:
        model.load_state_dict(sd, strict=True)
        print("[load_model] Weights loaded (strict=True)")
    except RuntimeError:
        mk = set(model.state_dict())
        ck = set(sd)
        missing    = sorted(mk - ck)
        unexpected = sorted(ck - mk)
        if missing:
            print(f"[load_model] Missing keys    ({len(missing)}): "
                  f"{missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[load_model] Unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
        model.load_state_dict(sd, strict=False)
        print("[load_model] Weights loaded (strict=False -- some layers randomised)")

    model.to(device).eval()

    print(f"[load_model] Model ready")
    print(f"  Classes     : {class_names if class_names else num_classes}")
    print(f"  fusion_dim  : {1536 + feat_out_dim}  "
          f"(1536 + feat_out_dim={feat_out_dim})")
    if val_acc  is not None: print(f"  Val Acc  : {val_acc:.2f}%")
    if macro_f1 is not None: print(f"  Macro-F1 : {macro_f1:.4f}")
    if epoch    is not None: print(f"  Epoch    : {epoch}")
    if is_swa:               print(f"  SWA      : Yes")

    return model


def get_class_names():
    return ['Dysplasia', 'Koilocytosis', 'Metaplasia', 'Parabasal', 'Superficial']


# ---------------------------------------------------------------------------
# Smoke test:  python load_model.py <checkpoint.pth>
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', nargs='?',
                   default='checkpoints/best_hybrid_model.pth')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    model = load_model(args.checkpoint, device=args.device)

    with torch.no_grad():
        logits, attn = model(torch.randn(1, 3, 300, 300), torch.randn(1, 30))

    import torch.nn.functional as F
    probs = F.softmax(logits, dim=-1)[0]
    for name, prob in zip(get_class_names(), probs.tolist()):
        print(f"  {name:<20} {prob:.4f}")
    print(f"\nSmoke test passed -- logits {logits.shape}  attn {attn.shape}")