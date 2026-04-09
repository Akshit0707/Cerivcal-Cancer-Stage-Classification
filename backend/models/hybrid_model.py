"""
CPU-Optimized Hybrid Model for Cervical Cancer Classification
Combines CNN features with traditional medical features for best accuracy on CPU

Architecture designed for:
- Efficient inference on CPU (no GPU required)
- High accuracy through multi-modal fusion
- Interpretability via attention mechanisms
- Medical domain knowledge integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class EfficientCNNBackbone(nn.Module):
    """
    Efficient CNN backbone optimized for CPU inference
    Uses depth-wise separable convolutions to reduce computation
    """
    def __init__(self, in_channels=3, base_channels=48):
        super(EfficientCNNBackbone, self).__init__()
        
        # Efficient stem - initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        
        # Efficient blocks using depthwise separable convolutions
        self.block1 = self._make_efficient_block(base_channels, base_channels * 2, stride=2)
        self.block2 = self._make_efficient_block(base_channels * 2, base_channels * 4, stride=2)
        self.block3 = self._make_efficient_block(base_channels * 4, base_channels * 8, stride=2)
        self.block4 = self._make_efficient_block(base_channels * 8, base_channels * 16, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate output features
        self.out_features = base_channels * 16
        
    def _make_efficient_block(self, in_channels, out_channels, stride=1):
        """
        Efficient block using depthwise separable convolution
        Significantly reduces parameters and computation
        """
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class FeatureAttention(nn.Module):
    """
    Attention mechanism to weight the importance of traditional features
    Helps the model focus on most relevant medical parameters
    """
    def __init__(self, num_features):
        super(FeatureAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(num_features)
        self.attention = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(num_features, num_features),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x_norm = self.layer_norm(x)
        attention_weights = self.attention(x_norm)
        return x * attention_weights, attention_weights


class CrossModalFusion(nn.Module):
    """
    Intelligent fusion of CNN features and traditional features
    Uses gating mechanism to balance contributions
    """
    def __init__(self, cnn_features, traditional_features, fusion_features=256):
        super(CrossModalFusion, self).__init__()
        
        # Project features to same dimension
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_features, fusion_features),
            nn.LayerNorm(fusion_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        
        self.traditional_projection = nn.Sequential(
            nn.Linear(traditional_features, fusion_features),
            nn.LayerNorm(fusion_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        
        # Gating mechanism to balance modalities
        self.gate = nn.Sequential(
            nn.Linear(fusion_features * 2, fusion_features),
            nn.Sigmoid()
        )
        
    def forward(self, cnn_feat, trad_feat):
        # Project to same dimension
        cnn_proj = self.cnn_projection(cnn_feat)
        trad_proj = self.traditional_projection(trad_feat)
        
        # Concatenate for gate
        combined = torch.cat([cnn_proj, trad_proj], dim=1)
        gate_weights = self.gate(combined)
        
        # Apply gating
        fused = gate_weights * cnn_proj + (1 - gate_weights) * trad_proj
        
        return fused


class CPUOptimizedHybridModel(nn.Module):
    """
    CPU-Optimized Hybrid Model combining CNN and traditional features
    
    Key features:
    - Efficient CNN backbone with depthwise separable convolutions
    - Feature attention for interpretability
    - Cross-modal fusion for optimal combination
    - Optimized for CPU inference with minimal latency
    
    Args:
        num_classes: Number of output classes (default: 5)
        num_traditional_features: Number of traditional features (default: 30)
        base_channels: Base number of channels for CNN (default: 32)
        fusion_features: Size of fusion layer (default: 256)
    """
    def __init__(
        self, 
        num_classes=5, 
        num_traditional_features=30,
        base_channels=48,
        fusion_features=256
    ):
        super(CPUOptimizedHybridModel, self).__init__()
        
        self.num_classes = num_classes
        self.num_traditional_features = num_traditional_features
        
        # CNN backbone for image features
        self.cnn_backbone = EfficientCNNBackbone(in_channels=3, base_channels=base_channels)
        cnn_out_features = self.cnn_backbone.out_features
        
        # Feature attention for traditional features
        self.feature_attention = FeatureAttention(num_traditional_features)
        
        # Cross-modal fusion
        self.fusion = CrossModalFusion(
            cnn_features=cnn_out_features,
            traditional_features=num_traditional_features,
            fusion_features=fusion_features
        )
        
        # Final classifier with better architecture
        self.classifier = nn.Sequential(
            nn.Linear(fusion_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights with proper scaling for final layer"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # CRITICAL: Scale down final layer by 0.01 to prevent extreme logits
        final_layer = self.classifier[-1]
        final_layer.weight.data *= 0.01
        final_layer.bias.data.zero_()
    
    def forward(self, image, traditional_features):
        """
        Forward pass
        
        Args:
            image: Input image tensor (B, 3, H, W)
            traditional_features: Traditional features tensor (B, num_features)
            
        Returns:
            logits: Class logits (B, num_classes)
            attention_weights: Feature attention weights for interpretability
        """
        # Extract CNN features
        cnn_features = self.cnn_backbone(image)
        
        # Apply attention to traditional features
        trad_features_attended, attention_weights = self.feature_attention(traditional_features)
        
        # Fuse features
        fused_features = self.fusion(cnn_features, trad_features_attended)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits, attention_weights
    
    def get_num_params(self):
        """Get number of parameters in model"""
        return sum(p.numel() for p in self.parameters())
    
    def get_feature_importance(self, attention_weights, feature_names):
        """
        Get feature importance scores for interpretability
        
        Args:
            attention_weights: Attention weights from forward pass
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Average across batch if needed
        if attention_weights.ndim > 1:
            attention_weights = attention_weights.mean(axis=0)
        
        importance_dict = {
            name: float(weight) 
            for name, weight in zip(feature_names, attention_weights)
        }
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True))
        
        return importance_dict


def create_hybrid_model(
    num_classes=5, 
    num_traditional_features=30,
    pretrained_cnn_path: Optional[str] = None,
    device='cpu'
):
    """
    Create CPU-optimized hybrid model
    
    Args:
        num_classes: Number of output classes
        num_traditional_features: Number of traditional features
        pretrained_cnn_path: Path to pretrained CNN weights (optional)
        device: Device to load model on
        
    Returns:
        Configured hybrid model
    """
    model = CPUOptimizedHybridModel(
        num_classes=num_classes,
        num_traditional_features=num_traditional_features,
        base_channels=32,  # Optimized for CPU
        fusion_features=256
    )
    
    # Load pretrained CNN weights if provided
    if pretrained_cnn_path is not None:
        try:
            print(f"Loading pretrained CNN backbone from {pretrained_cnn_path}...")
            # Note: This would need adaptation based on your CNN model structure
            # For now, we'll train from scratch for best CPU optimization
            print("Note: Training from scratch for optimal CPU performance")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            print("Training from scratch...")
    
    model = model.to(device)
    
    # Print model info
    num_params = model.get_num_params()
    print(f"\nCPU-Optimized Hybrid Model created:")
    print(f"  - Total parameters: {num_params:,}")
    print(f"  - Estimated size: {num_params * 4 / 1024 / 1024:.2f} MB")
    print(f"  - Device: {device}")
    print(f"  - Classes: {num_classes}")
    print(f"  - Traditional features: {num_traditional_features}")
    
    return model


def load_hybrid_model(checkpoint_path, device='cpu'):
    """
    Load a trained hybrid model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model configuration
    num_classes = checkpoint.get('num_classes', 5)
    num_traditional_features = checkpoint.get('num_traditional_features', 30)
    
    # Create model
    model = create_hybrid_model(
        num_classes=num_classes,
        num_traditional_features=num_traditional_features,
        device=device
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nModel loaded from {checkpoint_path}")
    if 'val_accuracy' in checkpoint:
        print(f"  - Validation accuracy: {checkpoint['val_accuracy']:.2f}%")
    if 'epoch' in checkpoint:
        print(f"  - Trained for {checkpoint['epoch']} epochs")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing CPU-Optimized Hybrid Model...")
    
    model = create_hybrid_model(
        num_classes=5,
        num_traditional_features=30,
        device='cpu'
    )
    
    # Test forward pass
    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_features = torch.randn(batch_size, 30)
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        logits, attention_weights = model(dummy_image, dummy_features)
    
    print(f"Input image shape: {dummy_image.shape}")
    print(f"Input features shape: {dummy_features.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Test feature importance
    feature_names = [f"feature_{i}" for i in range(30)]
    importance = model.get_feature_importance(attention_weights, feature_names)
    print("\nTop 5 most important features:")
    for i, (name, score) in enumerate(list(importance.items())[:5], 1):
        print(f"  {i}. {name}: {score:.4f}")
    
    print("\n✓ Model test completed successfully!")
