"""
Traditional ML Feature Extraction for Cervical Cancer Classification
Extracts hand-crafted features like cell size, shape, texture, etc.
"""
import cv2
import numpy as np
from skimage import feature, measure, morphology
from skimage.color import rgb2gray
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class CellFeatureExtractor:
    """Extract medical features from cervical cell images"""
    
    def __init__(self):
        pass
    
    def extract_morphological_features(self, image):
        """Extract cell shape and size features"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to segment cells
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours (cells)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = {}
        
        if len(contours) > 0:
            # Get largest contour (assume main cell)
            main_cell = max(contours, key=cv2.contourArea)
            
            # Cell area
            features['cell_area'] = cv2.contourArea(main_cell)
            
            # Cell perimeter
            features['cell_perimeter'] = cv2.arcLength(main_cell, True)
            
            # Compactness (circularity)
            if features['cell_perimeter'] > 0:
                features['compactness'] = (4 * np.pi * features['cell_area']) / (features['cell_perimeter'] ** 2)
            else:
                features['compactness'] = 0
            
            # Bounding box aspect ratio
            x, y, w, h = cv2.boundingRect(main_cell)
            features['aspect_ratio'] = float(w) / h if h > 0 else 0
            features['bbox_width'] = w
            features['bbox_height'] = h
            
            # Solidity (convexity)
            hull = cv2.convexHull(main_cell)
            hull_area = cv2.contourArea(hull)
            features['solidity'] = features['cell_area'] / hull_area if hull_area > 0 else 0
            
            # Extent (area ratio to bounding box)
            bbox_area = w * h
            features['extent'] = features['cell_area'] / bbox_area if bbox_area > 0 else 0
            
        else:
            # No cells found - use default values
            features['cell_area'] = 0
            features['cell_perimeter'] = 0
            features['compactness'] = 0
            features['aspect_ratio'] = 0
            features['bbox_width'] = 0
            features['bbox_height'] = 0
            features['solidity'] = 0
            features['extent'] = 0
        
        return features
    
    def extract_texture_features(self, image):
        """Extract texture features using Local Binary Patterns and GLCM"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        features = {}
        
        # Local Binary Pattern (LBP)
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # LBP histogram features
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        
        features['lbp_mean'] = np.mean(hist)
        features['lbp_std'] = np.std(hist)
        features['lbp_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Gray Level Co-occurrence Matrix (GLCM)
        # Resize for faster computation
        small_gray = cv2.resize(gray, (128, 128))
        glcm = feature.graycomatrix(small_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                     levels=256, symmetric=True, normed=True)
        
        # GLCM properties
        features['glcm_contrast'] = feature.graycoprops(glcm, 'contrast').mean()
        features['glcm_dissimilarity'] = feature.graycoprops(glcm, 'dissimilarity').mean()
        features['glcm_homogeneity'] = feature.graycoprops(glcm, 'homogeneity').mean()
        features['glcm_energy'] = feature.graycoprops(glcm, 'energy').mean()
        features['glcm_correlation'] = feature.graycoprops(glcm, 'correlation').mean()
        
        return features
    
    def extract_color_features(self, image):
        """Extract color-based features"""
        features = {}
        
        # Mean and std of each channel
        for i, channel in enumerate(['red', 'green', 'blue']):
            features[f'{channel}_mean'] = np.mean(image[:, :, i])
            features[f'{channel}_std'] = np.std(image[:, :, i])
        
        # HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        for i, channel in enumerate(['hue', 'saturation', 'value']):
            features[f'{channel}_mean'] = np.mean(hsv[:, :, i])
            features[f'{channel}_std'] = np.std(hsv[:, :, i])
        
        return features
    
    def extract_nucleus_features(self, image):
        """Extract nucleus-specific features"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Enhance nucleus (darker regions)
        nucleus = cv2.bitwise_not(gray)
        
        # Threshold to isolate nucleus
        _, nucleus_binary = cv2.threshold(nucleus, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find nucleus contours
        contours, _ = cv2.findContours(nucleus_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = {}
        
        if len(contours) > 0:
            nucleus_contour = max(contours, key=cv2.contourArea)
            
            features['nucleus_area'] = cv2.contourArea(nucleus_contour)
            features['nucleus_perimeter'] = cv2.arcLength(nucleus_contour, True)
            
            # Nucleus to cytoplasm ratio (approximation)
            total_area = image.shape[0] * image.shape[1]
            features['nucleus_cytoplasm_ratio'] = features['nucleus_area'] / (total_area - features['nucleus_area']) if total_area > features['nucleus_area'] else 0
            
            # Nucleus irregularity
            if features['nucleus_perimeter'] > 0:
                features['nucleus_irregularity'] = (4 * np.pi * features['nucleus_area']) / (features['nucleus_perimeter'] ** 2)
            else:
                features['nucleus_irregularity'] = 0
        else:
            features['nucleus_area'] = 0
            features['nucleus_perimeter'] = 0
            features['nucleus_cytoplasm_ratio'] = 0
            features['nucleus_irregularity'] = 0
        
        return features
    
    def extract_all_features(self, image):
        """Extract all features from an image"""
        features = {}
        
        # Resize image for consistent feature extraction
        image_resized = cv2.resize(image, (224, 224))
        
        # Extract different feature types
        features.update(self.extract_morphological_features(image_resized))
        features.update(self.extract_texture_features(image_resized))
        features.update(self.extract_color_features(image_resized))
        features.update(self.extract_nucleus_features(image_resized))
        
        return features


def extract_medical_features(image):
    """
    Extract medical features from a single PIL Image
    
    Args:
        image: PIL Image object
        
    Returns:
        List of 30 feature values
    """
    extractor = CellFeatureExtractor()
    
    # Convert PIL Image to numpy array
    if hasattr(image, 'convert'):
        image = np.array(image.convert('RGB'))
    
    # Extract all features
    features_dict = extractor.extract_all_features(image)
    
    # Return features in consistent order (30 features)
    feature_order = [
        'cell_area', 'cell_perimeter', 'compactness', 'aspect_ratio',
        'solidity', 'extent', 'nucleus_area', 'nucleus_cytoplasm_ratio',
        'nucleus_irregularity', 'lbp_entropy', 'lbp_mean', 'lbp_std',
        'glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation',
        'red_mean', 'green_mean', 'blue_mean', 'red_std', 'green_std', 'blue_std',
        'hue_mean', 'saturation_mean', 'value_mean', 'hue_std', 'saturation_std', 'value_std',
        'bbox_width', 'bbox_height'
    ]
    
    features = [features_dict.get(key, 0.0) for key in feature_order]
    return features


def extract_features_from_dataset(data_dir, output_csv='features.csv'):
    """Extract features from all images in dataset"""
    extractor = CellFeatureExtractor()
    data_dir = Path(data_dir)
    
    all_features = []
    classes = ['Normal', 'CIN1', 'CIN2', 'CIN3', 'Cancer']
    
    print("🔬 Extracting hand-crafted features from images...")
    
    for class_name in classes:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"⚠️  Warning: {class_dir} not found")
            continue
        
        image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
        print(f"\nProcessing {class_name}: {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc=f"  {class_name}"):
            try:
                # Read image
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Extract features
                features = extractor.extract_all_features(image)
                features['class'] = class_name
                features['filename'] = img_path.name
                
                all_features.append(features)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Save to CSV
    output_path = data_dir.parent / output_csv
    df.to_csv(output_path, index=False)
    print(f"\n✅ Features saved to: {output_path}")
    print(f"📊 Total samples: {len(df)}")
    print(f"📋 Total features: {len(df.columns) - 2}")  # -2 for class and filename
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract traditional ML features from cervical images')
    parser.add_argument('--data-dir', type=str, default='../data/train',
                        help='Directory containing training images')
    parser.add_argument('--output', type=str, default='train_features.csv',
                        help='Output CSV filename')
    
    args = parser.parse_args()
    
    extract_features_from_dataset(args.data_dir, args.output)
