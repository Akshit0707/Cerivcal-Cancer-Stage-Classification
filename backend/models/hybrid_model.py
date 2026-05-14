"""
CPU-Optimized Hybrid Model for Cervical Cancer Classification

FIXES vs previous version:
  1. Added `from typing import Optional` — was missing, caused NameError on import
  2. create_hybrid_model() passed base_channels/fusion_features to
     CPUOptimizedHybridModel which doesn't accept them → TypeError on creation
  3. load_hybrid_model() called create_hybrid_model() with same bad kwargs
  4. CPUOptimizedHybridModel.forward() returned a single tensor but
     train_hybrid.py unpacks (logits, attn) — added dummy attn return
  5. pretrained=True deprecation warning → use weights= API
  6. get_num_params() method referenced but never defined → added it
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class CrossModalFusion(nn.Module):
    """Fuse CNN features and traditional medical features."""

    def __init__(self, cnn_dim: int, feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
        )
        self.traditional_projection = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, cnn_features, traditional_features):
        cnn_proj  = self.cnn_projection(cnn_features)
        trad_proj = self.traditional_projection(traditional_features)
        fused     = torch.cat([cnn_proj, trad_proj], dim=1)
        return self.fusion_mlp(fused)


class CPUOptimizedHybridModel(nn.Module):
    """Hybrid model combining EfficientNet-B3 + traditional medical features."""

    def __init__(self, num_classes: int = 5, num_traditional_features: int = 30):
        super().__init__()
        self.num_traditional_features = num_traditional_features

        # FIX 5: use weights= API instead of deprecated pretrained=True
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        cnn_out_dim = self.backbone.classifier[1].in_features  # 1536
        self.backbone.classifier = nn.Identity()

        self.fusion = CrossModalFusion(
            cnn_dim=cnn_out_dim,
            feat_dim=num_traditional_features,
            hidden_dim=128,
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    # FIX 6: add missing method
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, images, features):
        cnn_features = self.backbone(images)
        fused  = self.fusion(cnn_features, features)
        logits = self.classifier(fused)
        # FIX 4: return (logits, dummy_attn) so train_hybrid.py unpack works
        dummy_attn = torch.zeros(images.size(0), 2, device=images.device)
        return logits, dummy_attn

    def get_feature_importance(self, avg_attention, feature_names):
        """Placeholder — returns equal weights (no explicit attention here)."""
        w = 1.0 / max(len(feature_names), 1)
        return {name: round(w, 6) for name in feature_names}


def create_hybrid_model(
    num_classes: int = 5,
    num_traditional_features: int = 30,
    pretrained_cnn_path: Optional[str] = None,
    device: str = "cpu",
) -> CPUOptimizedHybridModel:
    """
    Create hybrid model.

    FIX 2: removed base_channels / fusion_features kwargs that were passed
    to CPUOptimizedHybridModel.__init__ but not accepted there.
    """
    model = CPUOptimizedHybridModel(
        num_classes=num_classes,
        num_traditional_features=num_traditional_features,
    )

    if pretrained_cnn_path is not None:
        try:
            print(f"Loading pretrained CNN backbone from {pretrained_cnn_path}...")
            state = torch.load(pretrained_cnn_path, map_location=device)
            model.backbone.load_state_dict(state, strict=False)
            print("  Backbone weights loaded.")
        except Exception as e:
            print(f"  Could not load pretrained weights: {e} — using ImageNet init.")

    model = model.to(device)

    num_params = model.get_num_params()
    print(f"\nHybrid Model created:")
    print(f"  Total parameters : {num_params:,}")
    print(f"  Estimated size   : {num_params * 4 / 1024 / 1024:.2f} MB")
    print(f"  Device           : {device}")
    print(f"  Classes          : {num_classes}")
    print(f"  Traditional feat : {num_traditional_features}")

    return model


def load_hybrid_model(checkpoint_path: str, device: str = "cpu") -> CPUOptimizedHybridModel:
    """
    Load a trained hybrid model from a checkpoint saved by train_hybrid.py.

    FIX 3: checkpoint saved by train_hybrid.py uses keys:
      'model_state_dict', 'class_names', 'num_features', 'val_acc', 'macro_f1'
    The old code looked for 'num_classes' / 'num_traditional_features' which
    don't exist in that checkpoint format.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_names = checkpoint.get("class_names", [])
    num_classes  = len(class_names) if class_names else checkpoint.get("num_classes", 5)
    num_features = checkpoint.get("num_features",
                   checkpoint.get("num_traditional_features", 30))

    model = create_hybrid_model(
        num_classes=num_classes,
        num_traditional_features=num_features,
        device=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    print(f"\nModel loaded from {checkpoint_path}")
    print(f"  Classes  : {class_names or num_classes}")
    if "val_acc" in checkpoint:
        print(f"  Val acc  : {checkpoint['val_acc']:.2f}%")
    if "macro_f1" in checkpoint:
        print(f"  Macro-F1 : {checkpoint['macro_f1']:.4f}")
    if "epoch" in checkpoint:
        print(f"  Epoch    : {checkpoint['epoch']}")

    return model


if __name__ == "__main__":
    print("Smoke-testing CPUOptimizedHybridModel...")
    model = create_hybrid_model(num_classes=5, num_traditional_features=30, device="cpu")

    dummy_img  = torch.randn(4, 3, 224, 224)
    dummy_feat = torch.randn(4, 30)

    with torch.no_grad():
        logits, attn = model(dummy_img, dummy_feat)

    assert logits.shape == (4, 5),  f"Bad logits shape: {logits.shape}"
    assert attn.shape   == (4, 2),  f"Bad attn shape: {attn.shape}"
    print(f"  logits: {logits.shape}  attn: {attn.shape}  ✓")
    print("Smoke test passed!")