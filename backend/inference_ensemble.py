"""
Ensemble Inference Script
Combines predictions from multiple models for more robust results
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


def predict_ensemble(models, image_path, feature_scaler, device='cuda', use_tta=False):
    """
    Predict using ensemble of models
    
    Args:
        models: List of trained models
        image_path: Path to image
        feature_scaler: Fitted StandardScaler for features
        device: Device to run inference on
        use_tta: Whether to use test-time augmentation
    
    Returns:
        averaged_probs: Averaged probabilities across all models
        predicted_class: Final predicted class
        confidence: Confidence in prediction
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Extract features
    features = extract_medical_features(np.array(image))
    features = feature_scaler.transform([features])[0]
    features = np.clip(features, -10, 10)
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # Base transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # TTA transforms if enabled
    if use_tta:
        tta_transforms = [
            transform,
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        ]
    else:
        tta_transforms = [transform]
    
    # Collect predictions from all models and augmentations
    all_probs = []
    
    with torch.no_grad():
        for model in models:
            model.eval()
            for t in tta_transforms:
                img_tensor = t(image).unsqueeze(0).to(device)
                logits, _ = model(img_tensor, features_tensor)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy()[0])
    
    # Average all predictions
    averaged_probs = np.mean(all_probs, axis=0)
    predicted_class = np.argmax(averaged_probs)
    confidence = averaged_probs[predicted_class]
    
    return averaged_probs, predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(description='Ensemble Inference')
    parser.add_argument('--model-paths', type=str, nargs='+', required=True,
                       help='Paths to trained model checkpoints (space-separated)')
    parser.add_argument('--image-path', type=str, required=True,
                       help='Path to image or directory of images')
    parser.add_argument('--scaler-path', type=str, default='./feature_scaler.pkl',
                       help='Path to fitted feature scaler')
    parser.add_argument('--use-tta', action='store_true',
                       help='Use test-time augmentation (3 augmentations per model)')
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
    
    # Load models
    print(f"\nLoading {len(args.model_paths)} models...")
    models = []
    for model_path in args.model_paths:
        print(f"  Loading {model_path}...")
        model = load_hybrid_model(model_path, device=device)
        models.append(model)
    
    # Get class names from first checkpoint
    checkpoint = torch.load(args.model_paths[0], map_location=device, weights_only=False)
    class_names = ['CIN1', 'CIN2', 'CIN3', 'Cancer', 'Normal']
    if 'class_names' in checkpoint:
        class_names = checkpoint['class_names']
    
    # Load feature scaler
    print(f"\nLoading feature scaler from {args.scaler_path}...")
    feature_scaler = joblib.load(args.scaler_path)
    
    # Info
    tta_info = " with TTA (3 augmentations)" if args.use_tta else ""
    total_predictions = len(models) * (3 if args.use_tta else 1)
    print(f"\nEnsemble configuration:")
    print(f"  Models: {len(models)}")
    print(f"  TTA: {'Enabled' if args.use_tta else 'Disabled'}")
    print(f"  Total predictions per image: {total_predictions}")
    
    # Check if path is directory or single image
    image_path = Path(args.image_path)
    
    if image_path.is_dir():
        # Process all images in directory
        image_files = list(image_path.glob('*.png')) + list(image_path.glob('*.jpg'))
        print(f"\nProcessing {len(image_files)} images with ensemble{tta_info}...")
        
        results = []
        for img_file in tqdm(image_files):
            probs, pred_class, confidence = predict_ensemble(
                models, img_file, feature_scaler, device, args.use_tta
            )
            results.append({
                'image': img_file.name,
                'predicted_class': class_names[pred_class],
                'confidence': confidence,
                'probabilities': probs
            })
        
        # Print summary
        print("\n" + "="*80)
        print("Ensemble Inference Results")
        print("="*80)
        for result in results:
            print(f"\n{result['image']}:")
            print(f"  Prediction: {result['predicted_class']} ({result['confidence']:.2%})")
            print(f"  All probabilities:")
            for cls_name, prob in zip(class_names, result['probabilities']):
                print(f"    {cls_name}: {prob:.2%}")
    
    else:
        # Single image
        print(f"\nProcessing {image_path.name} with ensemble{tta_info}...")
        probs, pred_class, confidence = predict_ensemble(
            models, image_path, feature_scaler, device, args.use_tta
        )
        
        print("\n" + "="*80)
        print("Ensemble Inference Result")
        print("="*80)
        print(f"\nImage: {image_path.name}")
        print(f"Prediction: {class_names[pred_class]} ({confidence:.2%})")
        print(f"\nAll probabilities:")
        for cls_name, prob in zip(class_names, probs):
            print(f"  {cls_name}: {prob:.2%}")
    
    print("\n✓ Ensemble inference completed!")


if __name__ == '__main__':
    main()
