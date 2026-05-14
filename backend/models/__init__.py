import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import sys
import torch
import torch.nn as nn
import os


def load_model(checkpoint_path, num_classes=5, num_traditional_features=30, device='cpu'):
    """Load hybrid model from checkpoint with dynamic dimension inference."""
    # FIXED: #3 infer fusion_dim from checkpoint instead of hard-coding 1664
    
    from hybrid_model import CPUOptimizedHybridModel
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model (dummy initialization)
    model = CPUOptimizedHybridModel(num_classes=num_classes, 
                                     num_traditional_features=num_traditional_features)
    
    # Infer attention input dimension from checkpoint
    # FIXED: #3 read attention layer shape from saved state dict
    if 'attention.0.weight' in checkpoint:
        attn_in_dim = checkpoint['attention.0.weight'].shape[1]
    else:
        # Fallback: compute from CNN and feature dims
        cnn_dim = 1536  # EfficientNet-B3
        attn_in_dim = 128  # fusion output (hardcoded in CrossModalFusion)
    
    # Rebuild attention layer with correct dimensions
    # FIXED: #3 use inferred dimension instead of hard-coded 1664
    model.attention = nn.Sequential(
        nn.Linear(attn_in_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, attn_in_dim),
        nn.Sigmoid()
    )
    
    # Load state dict
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def get_class_names():
    """Return class names for cervical cancer classification."""
    return ['Dysplasia', 'Koilocytosis', 'Metaplasia', 'Parabasal', 'Superficial']
