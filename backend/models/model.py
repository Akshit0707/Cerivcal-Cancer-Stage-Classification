"""
Hybrid Model for Cervical Cancer Classification
Architecture reverse-engineered from actual checkpoint state-dict shapes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERIFIED checkpoint architecture (all dims read from weight shapes):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  FeatureMLP  (3-block, indices 0-9 in net):
    net.0  Linear(30  -> 256)   [256, 30]
    net.1  BN(256)
    net.2  GELU
    net.3  Dropout(0.3)
    net.4  Linear(256 -> 256)   [256, 256]
    net.5  BN(256)
    net.6  GELU
    net.7  Dropout(0.15)
    net.8  Linear(256 -> 128)   [128, 256]   <- final reduction
    net.9  BN(128)
    proj   Linear(30  -> 128)   [128, 30]
    forward: net(x) + proj(x)  -> 128-dim

  fusion_dim = 1536 (EfficientNet-B3) + 128 = 1664

  attention  (NO Dropout, indices 0-3):
    attention.0  Linear(1664 -> 64)
    attention.1  GELU
    attention.2  Linear(64   -> 2)
    attention.3  Softmax(dim=-1)

  classifier (indices 0-5):
    classifier.0  Dropout(0.5)
    classifier.1  Linear(1664 -> 512)
    classifier.2  BN(512)
    classifier.3  GELU
    classifier.4  Dropout(0.25)
    classifier.5  Linear(512 -> num_classes)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# FeatureMLP  -- 3-block, verified against checkpoint
# ---------------------------------------------------------------------------
class FeatureMLP(nn.Module):
    """
    Three-linear-block MLP with a residual projection.

    Default dims match the saved checkpoint:
      in_dim=30, hidden1=256, hidden2=256, out_dim=128

    Sequential index map (state-dict key alignment):
      net.0  Linear(in_dim  -> hidden1)
      net.1  BN(hidden1)
      net.2  GELU
      net.3  Dropout(dropout)
      net.4  Linear(hidden1 -> hidden2)
      net.5  BN(hidden2)
      net.6  GELU
      net.7  Dropout(dropout * 0.5)
      net.8  Linear(hidden2 -> out_dim)
      net.9  BN(out_dim)
      proj   Linear(in_dim  -> out_dim)   [always, since in_dim != out_dim]
    """

    def __init__(
        self,
        in_dim:  int   = 30,
        out_dim: int   = 128,
        hidden1: int   = 256,
        hidden2: int   = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,  hidden1),   # net.0
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
        self.proj = (
            nn.Linear(in_dim, out_dim)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.proj(x)


# ---------------------------------------------------------------------------
# DropPath  -- used only in _inject_drop_path
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rt = torch.rand(shape, dtype=x.dtype, device=x.device)
        return x / keep_prob * torch.floor(rt + keep_prob)


# ---------------------------------------------------------------------------
# EfficientNetHybrid
# ---------------------------------------------------------------------------
class EfficientNetHybrid(nn.Module):
    """
    EfficientNet-B3 backbone + 3-block FeatureMLP, fused via attention.
    All dims default to the saved checkpoint values.
    """

    def __init__(
        self,
        num_classes:    int   = 5,
        num_features:   int   = 30,
        feat_out_dim:   int   = 128,   # FeatureMLP output dim
        mlp_hidden1:    int   = 256,
        mlp_hidden2:    int   = 256,
        attn_hidden:    int   = 64,
        attn_dropout:   bool  = False, # True adds Dropout(0.1) inside attention
        dropout:        float = 0.5,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.num_features = num_features

        # -- Backbone -------------------------------------------------------
        backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        self.cnn_out_dim = backbone.classifier[1].in_features  # 1536
        backbone.classifier = nn.Identity()
        self._inject_drop_path(backbone, drop_path_rate)
        self.backbone = backbone

        # -- FeatureMLP -----------------------------------------------------
        self.feature_mlp = FeatureMLP(
            in_dim  = num_features,
            out_dim = feat_out_dim,
            hidden1 = mlp_hidden1,
            hidden2 = mlp_hidden2,
            dropout = 0.3,
        )

        fusion_dim = self.cnn_out_dim + feat_out_dim  # 1536 + 128 = 1664

        # -- Attention gate -------------------------------------------------
        # Checkpoint has NO Dropout: Linear->GELU->Linear->Softmax (idx 0-3)
        # attn_dropout=True adds Dropout at idx 2, shifting Linear to idx 3
        if attn_dropout:
            self.attention = nn.Sequential(
                nn.Linear(fusion_dim, attn_hidden),  # .0
                nn.GELU(),                            # .1
                nn.Dropout(0.1),                      # .2
                nn.Linear(attn_hidden, 2),            # .3
                nn.Softmax(dim=-1),                   # .4
            )
        else:
            self.attention = nn.Sequential(
                nn.Linear(fusion_dim, attn_hidden),  # .0
                nn.GELU(),                            # .1
                nn.Linear(attn_hidden, 2),            # .2
                nn.Softmax(dim=-1),                   # .3
            )

        # -- Classifier -----------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),           # .0
            nn.Linear(fusion_dim, 512),    # .1
            nn.BatchNorm1d(512),           # .2
            nn.GELU(),                     # .3
            nn.Dropout(dropout * 0.5),     # .4
            nn.Linear(512, num_classes),   # .5
        )

    def _inject_drop_path(self, backbone, rate: float) -> None:
        try:
            blocks = list(backbone.features.children())
            n = sum(1 for s in blocks if hasattr(s, '__iter__') for _ in s)
            dp_rates = torch.linspace(0, rate, max(n, 1)).tolist()
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

    def forward(self, images: torch.Tensor, features: torch.Tensor):
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


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def create_hybrid_model(
    num_classes:              int   = 5,
    num_traditional_features: int   = 30,
    feat_out_dim:             int   = 128,
    mlp_hidden1:              int   = 256,
    mlp_hidden2:              int   = 256,
    attn_hidden:              int   = 64,
    attn_dropout:             bool  = False,
    dropout:                  float = 0.5,
    drop_path_rate:           float = 0.2,
    device:                   str   = "cpu",
) -> EfficientNetHybrid:
    model = EfficientNetHybrid(
        num_classes    = num_classes,
        num_features   = num_traditional_features,
        feat_out_dim   = feat_out_dim,
        mlp_hidden1    = mlp_hidden1,
        mlp_hidden2    = mlp_hidden2,
        attn_hidden    = attn_hidden,
        attn_dropout   = attn_dropout,
        dropout        = dropout,
        drop_path_rate = drop_path_rate,
    ).to(device)

    n = sum(p.numel() for p in model.parameters())
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fusion = 1536 + feat_out_dim
    print(f"\nHybrid Model created:")
    print(f"  FeatureMLP  : {num_traditional_features}->{mlp_hidden1}->{mlp_hidden2}->{feat_out_dim}")
    print(f"  fusion_dim  : 1536 + {feat_out_dim} = {fusion}")
    print(f"  attn_hidden : {attn_hidden}  attn_dropout={attn_dropout}")
    print(f"  Total params: {n:,}  ({n*4/1024/1024:.1f} MB)  trainable: {t:,}")
    print(f"  Device      : {device}  classes: {num_classes}")
    return model


def load_hybrid_model(
    checkpoint_path: str,
    device:          str   = "cpu",
    dropout:         float = 0.5,
    drop_path_rate:  float = 0.2,
) -> EfficientNetHybrid:
    """Thin wrapper -- delegates to load_model.py for full dim-inference."""
    import importlib, sys
    from pathlib import Path
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    lm = importlib.import_module("load_model")
    return lm.load_model(checkpoint_path, device=device)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Smoke-test: EfficientNetHybrid (checkpoint-verified dims)")
    print("=" * 60)

    model = create_hybrid_model(num_classes=5, device="cpu")
    model.eval()

    with torch.no_grad():
        logits, attn = model(torch.randn(2, 3, 300, 300), torch.randn(2, 30))

    assert logits.shape == (2, 5)
    assert attn.shape   == (2, 2)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attn).all()

    sd = model.state_dict()
    checks = {
        "feature_mlp.net.0.weight": (256, 30),   # Linear(30->256)
        "feature_mlp.net.4.weight": (256, 256),  # Linear(256->256)
        "feature_mlp.net.8.weight": (128, 256),  # Linear(256->128)
        "feature_mlp.net.9.weight": (128,),       # BN(128)
        "feature_mlp.proj.weight":  (128, 30),   # proj
        "attention.0.weight":       (64, 1664),
        "attention.2.weight":       (2,  64),
        "classifier.1.weight":      (512, 1664),
        "classifier.5.weight":      (5,   512),
    }
    all_ok = True
    for key, exp in checks.items():
        act = tuple(sd[key].shape)
        ok  = act == exp
        all_ok = all_ok and ok
        print(f"  {'OK' if ok else 'FAIL'} {key}: {act}")

    print(f"\nForward OK: logits={logits.shape}  attn={attn.shape}")
    print("All checks passed." if all_ok else "SOME CHECKS FAILED.")