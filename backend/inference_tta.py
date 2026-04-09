"""
Test-Time Augmentation (TTA) Inference Script
Averages predictions over multiple augmented versions of each image
for more robust and accurate predictions
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
import argparse
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from models.hybrid_model import load_hybrid_model
from feature_extractor import extract_medical_features
from sklearn.preprocessing import StandardScaler
import joblib


def get_tta_transforms(num_augmentations=10):
    """Create multiple augmented versions of the same image"""
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Different augmentation combinations
    tta_transforms = [base_transform]  # Original
    
    for _ in range(num_augmentations - 1):
        tta_transforms.append(transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]))
    
    return tta_transforms


def predict_with_tta(model, image_path, feature_scaler, num_augmentations=10, device='cuda'):
    """
    Predict with Test-Time Augmentation
    
    Args:
        model: Trained hybrid model
        image_path: Path to image
        feature_scaler: Fitted StandardScaler for features
        num_augmentations: Number of augmented versions to average
        device: Device to run inference on
    
    Returns:
        averaged_probs: Averaged probabilities across all augmentations
        predicted_class: Final predicted class
        confidence: Confidence in prediction
    """
    model.eval()
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Extract features once (same for all augmentations)
    features = extract_medical_features(np.array(image))
    features = feature_scaler.transform([features])[0]
    features = np.clip(features, -10, 10)  # Clip extreme values
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # Get TTA transforms
    tta_transforms = get_tta_transforms(num_augmentations)
    
    # Collect predictions from all augmentations
    all_probs = []
    
    with torch.no_grad():
        for transform in tta_transforms:
            # Apply transform
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # Forward pass
            logits, _ = model(img_tensor, features_tensor)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy()[0])
    
    # Average all predictions
    averaged_probs = np.mean(all_probs, axis=0)
    predicted_class = np.argmax(averaged_probs)
    confidence = averaged_probs[predicted_class]
    
    return averaged_probs, predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(description='Inference with Test-Time Augmentation')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image-path', type=str, required=True,
                       help='Path to image or directory of images')
    parser.add_argument('--scaler-path', type=str, default='./feature_scaler.pkl',
                       help='Path to fitted feature scaler')
    parser.add_argument('--num-augmentations', type=int, default=10,
                       help='Number of augmentations for TTA')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = load_hybrid_model(args.model_path, device=device)
    
    # Get class names from checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    class_names = ['CIN1', 'CIN2', 'CIN3', 'Cancer', 'Normal']  # Default order
    if 'class_names' in checkpoint:
        class_names = checkpoint['class_names']
    
    # Load feature scaler
    print(f"Loading feature scaler from {args.scaler_path}...")
    feature_scaler = joblib.load(args.scaler_path)
    
    # Check if path is directory or single image
    image_path = Path(args.image_path)
    
    if image_path.is_dir():
        # Process all images in directory
        image_files = list(image_path.glob('*.png')) + list(image_path.glob('*.jpg'))
        print(f"\nProcessing {len(image_files)} images with TTA ({args.num_augmentations} augmentations each)...")
        
        results = []
        for img_file in tqdm(image_files):
            probs, pred_class, confidence = predict_with_tta(
                model, img_file, feature_scaler, args.num_augmentations, device
            )
            results.append({
                'image': img_file.name,
                'predicted_class': class_names[pred_class],
                'confidence': confidence,
                'probabilities': probs
            })
        
        # Print summary
        print("\n" + "="*80)
        print("TTA Inference Results")
        print("="*80)
        for result in results:
            print(f"\n{result['image']}:")
            print(f"  Prediction: {result['predicted_class']} ({result['confidence']:.2%})")
            print(f"  All probabilities:")
            for i, (cls_name, prob) in enumerate(zip(class_names, result['probabilities'])):
                print(f"    {cls_name}: {prob:.2%}")
    
    else:
        # Single image
        print(f"\nProcessing {image_path.name} with TTA ({args.num_augmentations} augmentations)...")
        probs, pred_class, confidence = predict_with_tta(
            model, image_path, feature_scaler, args.num_augmentations, device
        )
        
        print("\n" + "="*80)
        print("TTA Inference Result")
        print("="*80)
        print(f"\nImage: {image_path.name}")
        print(f"Prediction: {class_names[pred_class]} ({confidence:.2%})")
        print(f"\nAll probabilities:")
        for cls_name, prob in zip(class_names, probs):
            print(f"  {cls_name}: {prob:.2%}")
    
    print("\n✓ Inference completed!")


if __name__ == '__main__':
    main()
