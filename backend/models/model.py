"""
model.py — EfficientNetHybrid v11
Architecture aligned with train_hybrid.py (SIPaKMeD + Herlev | 4-class ordinal).

Key architecture facts
───────────────────────────────────────────────────────────────────────
  NUM_FEATURES  = 31  (30 medical dims + 1 synthetic flag)
  INPUT_SIZE    = 300
  FEAT_DIM      = 128  (FeatureMLP output)
  CNN_DIM       = 1536 (EfficientNet-B3 pool output)
  FUSION_DIM    = 1664 (CNN_DIM + FEAT_DIM)

  FeatureMLP  (gate-gated, state-dict prefix: feat_mlp)
    net.0  Linear(31  -> 256)
    net.1  BN(256)
    net.2  GELU
    net.3  Dropout(0.25)
    net.4  Linear(256 -> 256)
    net.5  BN(256)
    net.6  GELU
    net.7  Dropout(0.125)
    net.8  Linear(256 -> 128)
    net.9  BN(128)
    gate.0 Linear(31  -> 32)
    gate.1 ReLU
    gate.2 Linear(32  -> 1)
    gate.3 Sigmoid
    forward: net(x) * gate(x)   ← gated, NOT residual

  ChannelSE  (state-dict prefix: se)
    fc.0  Linear(1664 -> 104)  [max(8, 1664//16)]
    fc.1  ReLU
    fc.2  Linear(104  -> 1664)
    fc.3  Sigmoid
    forward: x * fc(x)

  head  (3-linear, state-dict prefix: head)
    head.0  Dropout(0.5)
    head.1  Linear(1664 -> 512)
    head.2  BN(512)
    head.3  GELU
    head.4  Dropout(0.25)
    head.5  Linear(512  -> 256)
    head.6  BN(256)
    head.7  GELU
    head.8  Dropout(0.125)
    head.9  Linear(256  -> num_classes)

  ordinal_head  (state-dict prefix: ordinal_head)
    ordinal_head.0  Dropout(0.25)
    ordinal_head.1  Linear(1664 -> 128)
    ordinal_head.2  GELU
    ordinal_head.3  Linear(128  -> num_classes-1)

  forward(images, features, return_ordinal=True):
    if return_ordinal: return logits, ord_logits
    else:              return logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ── Shared constants (mirror train_hybrid.py) ─────────────────────────────────
NUM_FEATURES   = 31     # 30 medical + 1 synthetic flag
INPUT_SIZE     = 300
FEAT_DIM       = 128
CNN_DIM        = 1536   # EfficientNet-B3 pool output
FUSION_DIM     = CNN_DIM + FEAT_DIM   # 1664
SEVERITY_ORDER = ['Normal', 'CIN1', 'HighGrade', 'Cancer']


# ── FeatureMLP ────────────────────────────────────────────────────────────────

class FeatureMLP(nn.Module):
    """
    Gate-gated MLP: output = net(x) * gate(x)

    State-dict key map (matches train_hybrid.py exactly):
      net.0  Linear(in_dim -> 256)
      net.1  BN(256)
      net.2  GELU
      net.3  Dropout(dropout)          [0.25 by default]
      net.4  Linear(256 -> 256)
      net.5  BN(256)
      net.6  GELU
      net.7  Dropout(dropout*0.5)      [0.125]
      net.8  Linear(256 -> out_dim)
      net.9  BN(out_dim)
      gate.0 Linear(in_dim -> 32)
      gate.1 ReLU
      gate.2 Linear(32 -> 1)
      gate.3 Sigmoid
    """

    def __init__(self, in_dim: int = 31, out_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),     # net.0
            nn.BatchNorm1d(256),        # net.1
            nn.GELU(),                  # net.2
            nn.Dropout(dropout),        # net.3
            nn.Linear(256, 256),        # net.4
            nn.BatchNorm1d(256),        # net.5
            nn.GELU(),                  # net.6
            nn.Dropout(dropout * 0.5),  # net.7
            nn.Linear(256, out_dim),    # net.8
            nn.BatchNorm1d(out_dim),    # net.9
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 32),      # gate.0
            nn.ReLU(),                  # gate.1
            nn.Linear(32, 1),           # gate.2
            nn.Sigmoid(),               # gate.3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.gate(x)


# ── ChannelSE ─────────────────────────────────────────────────────────────────

class ChannelSE(nn.Module):
    """
    Squeeze-and-Excitation on the fused 1-D feature vector.

    State-dict key map:
      fc.0  Linear(dim -> max(8, dim//r))
      fc.1  ReLU
      fc.2  Linear(max(8, dim//r) -> dim)
      fc.3  Sigmoid
    """

    def __init__(self, dim: int, r: int = 16):
        super().__init__()
        mid = max(8, dim // r)
        self.fc = nn.Sequential(
            nn.Linear(dim, mid),   # fc.0
            nn.ReLU(inplace=True), # fc.1
            nn.Linear(mid, dim),   # fc.2
            nn.Sigmoid(),          # fc.3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


# ── EfficientNetHybrid ────────────────────────────────────────────────────────

class EfficientNetHybrid(nn.Module):
    """
    EfficientNet-B3 backbone + FeatureMLP fused via ChannelSE,
    with a 3-linear classification head and a separate ordinal head.

    Matches train_hybrid.py v11 exactly.
    """

    def __init__(
        self,
        num_classes:    int   = 4,    # 4-class ordinal: Normal/CIN1/HighGrade/Cancer
        num_features:   int   = NUM_FEATURES,   # 31
        feat_dim:       int   = FEAT_DIM,       # 128
        dropout:        float = 0.3,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.num_features = num_features

        # ── Backbone ───────────────────────────────────────────────────────
        try:
            from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
            bb = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.cnn_dim = bb.classifier[1].in_features   # 1536
            bb.classifier = nn.Identity()
            self._set_drop_path(bb, drop_path_rate)
            self.backbone = bb
        except Exception:
            # Fallback to ResNet-50 if EfficientNet unavailable
            from torchvision.models import resnet50, ResNet50_Weights
            bb = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.cnn_dim = bb.fc.in_features
            bb.fc = nn.Identity()
            self.backbone = bb

        # ── Feature MLP (note: attribute name is feat_mlp) ────────────────
        self.feat_mlp = FeatureMLP(num_features, feat_dim, dropout=0.25)

        fusion = self.cnn_dim + feat_dim   # 1536 + 128 = 1664

        # ── Channel Squeeze-and-Excitation ────────────────────────────────
        self.se = ChannelSE(fusion, r=16)

        # ── Classification head (3 linear layers) ─────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),           # head.0
            nn.Linear(fusion, 512),        # head.1
            nn.BatchNorm1d(512),           # head.2
            nn.GELU(),                     # head.3
            nn.Dropout(dropout * 0.5),     # head.4
            nn.Linear(512, 256),           # head.5
            nn.BatchNorm1d(256),           # head.6
            nn.GELU(),                     # head.7
            nn.Dropout(dropout * 0.25),    # head.8
            nn.Linear(256, num_classes),   # head.9
        )

        # ── Ordinal head (parallel branch) ────────────────────────────────
        self.ordinal_head = nn.Sequential(
            nn.Dropout(dropout * 0.5),         # ordinal_head.0
            nn.Linear(fusion, 128),            # ordinal_head.1
            nn.GELU(),                         # ordinal_head.2
            nn.Linear(128, num_classes - 1),   # ordinal_head.3
        )

        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────────────────
    def _init_weights(self):
        for m in list(self.head.modules()) + list(self.ordinal_head.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Stochastic depth ──────────────────────────────────────────────────
    def _set_drop_path(self, bb, rate: float) -> None:
        try:
            blocks = list(bb.features.children())
            n = sum(1 for s in blocks if hasattr(s, '__iter__') for _ in s)
            rates = torch.linspace(0, rate, max(n, 1)).tolist()
            i = 0
            for s in blocks:
                if not hasattr(s, '__iter__'):
                    continue
                for b in s:
                    if hasattr(b, 'stochastic_depth'):
                        b.stochastic_depth.p = rates[i]
                    i += 1
        except Exception:
            pass

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(
        self,
        images:         torch.Tensor,
        features:       torch.Tensor,
        return_ordinal: bool = True,
    ):
        """
        Args:
            images        : FloatTensor [B, 3, 300, 300]
            features      : FloatTensor [B, 31]
            return_ordinal: if True  → return (logits, ord_logits)
                            if False → return logits only
        """
        images   = torch.nan_to_num(images,   nan=0., posinf=1.,  neginf=-1.)
        features = torch.nan_to_num(features, nan=0., posinf=0.,  neginf=0.)

        cnn  = torch.nan_to_num(self.backbone(images),
                                nan=0., posinf=1e3, neginf=-1e3)
        feat = torch.nan_to_num(self.feat_mlp(features),
                                nan=0., posinf=1e3, neginf=-1e3)

        fused  = self.se(torch.cat([cnn, feat], dim=1))
        logits = self.head(fused)

        if return_ordinal:
            return logits, self.ordinal_head(fused)
        return logits


# ── Public helpers ────────────────────────────────────────────────────────────

def create_hybrid_model(
    num_classes:    int   = 4,
    num_features:   int   = NUM_FEATURES,
    feat_dim:       int   = FEAT_DIM,
    dropout:        float = 0.5,
    drop_path_rate: float = 0.4,
    device:         str   = 'cpu',
) -> EfficientNetHybrid:
    """
    Instantiate and print an EfficientNetHybrid.
    Defaults match the values used in train_hybrid.py's build_model() call:
      dropout=0.5, dpr=0.4
    """
    model = EfficientNetHybrid(
        num_classes    = num_classes,
        num_features   = num_features,
        feat_dim       = feat_dim,
        dropout        = dropout,
        drop_path_rate = drop_path_rate,
    ).to(device)

    n = sum(p.numel() for p in model.parameters())
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fusion = model.cnn_dim + feat_dim
    print(f"\nEfficientNetHybrid v11 created:")
    print(f"  FeatureMLP  : {num_features} → 256 → 256 → {feat_dim}  (gate-gated)")
    print(f"  fusion_dim  : {model.cnn_dim} + {feat_dim} = {fusion}")
    print(f"  head        : {fusion} → 512 → 256 → {num_classes}")
    print(f"  ordinal_head: {fusion} → 128 → {num_classes - 1}")
    print(f"  Total params: {n:,}  ({n * 4 / 1024 / 1024:.1f} MB)  trainable: {t:,}")
    print(f"  Device      : {device}  classes: {num_classes}")
    return model


def load_hybrid_model(
    checkpoint_path: str,
    device:          str = 'cpu',
) -> EfficientNetHybrid:
    """Thin wrapper — delegates to load_model.py for full dim-inference."""
    import importlib, sys
    from pathlib import Path
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    lm = importlib.import_module('load_model')
    return lm.load_model(checkpoint_path, device=device)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('Smoke-test: EfficientNetHybrid v11')
    print(f'  NUM_FEATURES={NUM_FEATURES}  INPUT_SIZE={INPUT_SIZE}')
    print(f'  SEVERITY_ORDER={SEVERITY_ORDER}')
    print('=' * 60)

    model = create_hybrid_model(num_classes=4, device='cpu')
    model.eval()

    imgs  = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    feats = torch.randn(2, NUM_FEATURES)

    with torch.no_grad():
        logits, ord_logits = model(imgs, feats, return_ordinal=True)
        logits_only        = model(imgs, feats, return_ordinal=False)

    assert logits.shape      == (2, 4),  f"logits shape {logits.shape}"
    assert ord_logits.shape  == (2, 3),  f"ord_logits shape {ord_logits.shape}"
    assert logits_only.shape == (2, 4),  f"logits_only shape {logits_only.shape}"
    assert torch.isfinite(logits).all()
    assert torch.isfinite(ord_logits).all()

    sd = model.state_dict()
    checks = {
        # FeatureMLP
        'feat_mlp.net.0.weight':      (256, NUM_FEATURES),
        'feat_mlp.net.4.weight':      (256, 256),
        'feat_mlp.net.8.weight':      (FEAT_DIM, 256),
        'feat_mlp.net.9.weight':      (FEAT_DIM,),
        'feat_mlp.gate.0.weight':     (32, NUM_FEATURES),
        'feat_mlp.gate.2.weight':     (1, 32),
        # ChannelSE
        'se.fc.0.weight':             (max(8, FUSION_DIM // 16), FUSION_DIM),
        'se.fc.2.weight':             (FUSION_DIM, max(8, FUSION_DIM // 16)),
        # Classification head
        'head.1.weight':              (512, FUSION_DIM),
        'head.5.weight':              (256, 512),
        'head.9.weight':              (4,   256),
        # Ordinal head
        'ordinal_head.1.weight':      (128, FUSION_DIM),
        'ordinal_head.3.weight':      (3,   128),
    }

    all_ok = True
    for key, exp in checks.items():
        act = tuple(sd[key].shape)
        ok  = act == exp
        all_ok = all_ok and ok
        print(f"  {'OK  ' if ok else 'FAIL'} {key}: got {act}"
              + ('' if ok else f'  expected {exp}'))

    print(f'\nForward OK:')
    print(f'  logits      : {logits.shape}')
    print(f'  ord_logits  : {ord_logits.shape}')
    print(f'  logits_only : {logits_only.shape}')
    print('All checks passed.' if all_ok else '\n❌  SOME CHECKS FAILED.')