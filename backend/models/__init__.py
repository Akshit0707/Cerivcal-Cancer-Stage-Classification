import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import sys
import torch
import torch.nn as nn
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from .cnn_model import get_class_names

def load_model(model_path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    sd = checkpoint["model_state_dict"]

    from train_hybrid import EfficientNetHybrid

    # Read exact attention dims from checkpoint
    fusion_dim  = sd["attention.0.weight"].shape[1]
    attn_hidden = sd["attention.0.weight"].shape[0]
    print(f"[load_model] checkpoint attention: fusion_dim={fusion_dim} hidden={attn_hidden}")

    # Build model
    model = EfficientNetHybrid(num_classes=5, num_features=30)

    # Overwrite attention to match checkpoint EXACTLY
    model.attention = nn.Sequential(
        nn.Linear(fusion_dim, attn_hidden),
        nn.ReLU(inplace=True),
        nn.Linear(attn_hidden, 2),
        nn.Softmax(dim=-1),
    )

    # Verify shapes match before loading
    print(f"[load_model] model attention.0 shape: {model.attention[0].weight.shape}")
    print(f"[load_model] checkpoint attention.0 shape: {sd['attention.0.weight'].shape}")

    result = model.load_state_dict(sd, strict=False)
    print(f"[load_model] missing : {result.missing_keys}")
    print(f"[load_model] unexpected: {result.unexpected_keys}")

    model.to(device)
    model.eval()
    print("[load_model] SUCCESS")
    return model

__all__ = ["load_model", "get_class_names"]
