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

    model = EfficientNetHybrid(num_classes=5, num_features=30)

    # Patch attention to EXACTLY match checkpoint:
    # index 0 = Linear(1664,64)
    # index 1 = ReLU        (no params)
    # index 2 = Linear(64,2)
    # index 3 = Softmax      (no params)
    model.attention = nn.Sequential(
        nn.Linear(1664, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 2),
        nn.Softmax(dim=-1),
    )

    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    print("[load_model] SUCCESS")
    return model

__all__ = ["load_model", "get_class_names"]
