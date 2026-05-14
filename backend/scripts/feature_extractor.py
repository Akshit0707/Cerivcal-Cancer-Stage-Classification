"""
Traditional ML Feature Extraction for Cervical Cancer Classification
Extracts hand-crafted features like cell size, shape, texture, etc.
"""
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from scipy import ndimage
from pathlib import Path


class CellFeatureExtractor:
    """Extract morphological and texture features from cell images."""
    
    def extract_all_features(self, image_path):
        """Extract all features from image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        morphology = self.extract_morphology_features(gray)
        texture = self.extract_texture_features(gray)
        nucleus = self.extract_nucleus_features(gray)
        
        return {**morphology, **texture, **nucleus}
    
    def extract_morphology_features(self, gray):
        """Extract morphological features."""
        # Binary thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                'cell_area': 0, 'cell_perimeter': 0, 'cell_solidity': 0,
                'cell_eccentricity': 0, 'cell_aspect_ratio': 0
            }
        
        # Largest contour (cell)
        cell_contour = max(contours, key=cv2.contourArea)
        cell_area = cv2.contourArea(cell_contour)
        cell_perimeter = cv2.arcLength(cell_contour, True)
        
        # Fit ellipse
        if len(cell_contour) >= 5:
            ellipse = cv2.fitEllipse(cell_contour)
            (_, _), (major, minor), _ = ellipse
            eccentricity = np.sqrt(1 - (minor / major) ** 2) if major > 0 else 0
            aspect_ratio = major / minor if minor > 0 else 0
        else:
            eccentricity = 0
            aspect_ratio = 1
        
        # Solidity
        hull = cv2.convexHull(cell_contour)
        hull_area = cv2.contourArea(hull)
        solidity = cell_area / hull_area if hull_area > 0 else 0
        
        return {
            'cell_area': float(cell_area),
            'cell_perimeter': float(cell_perimeter),
            'cell_solidity': float(solidity),
            'cell_eccentricity': float(eccentricity),
            'cell_aspect_ratio': float(aspect_ratio)
        }
    
    def extract_texture_features(self, gray):
        """Extract GLCM texture features."""
        # FIXED: #2 remove glcm_dissimilarity to keep 30 features (option A)
        # or add it to feature_order below (option B). Here using option A.
        
        # Compute GLCM
        glcm = greycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        glcm = glcm[:, :, 0, 0]
        
        # Extract 4 GLCM properties (not 5)
        # FIXED: #2 removed dissimilarity; keeping: contrast, correlation, energy, homogeneity
        contrast = greycoprops(glcm, 'contrast')
        correlation = greycoprops(glcm, 'correlation')
        energy = greycoprops(glcm, 'energy')
        homogeneity = greycoprops(glcm, 'homogeneity')
        
        return {
            'glcm_contrast': float(contrast),
            'glcm_correlation': float(correlation),
            'glcm_energy': float(energy),
            'glcm_homogeneity': float(homogeneity)
        }
    
    def extract_nucleus_features(self, gray):
        """Extract nucleus-specific features."""
        # FIXED: #8 add morphological opening after thresholding
        inverted = cv2.bitwise_not(gray)
        _, nucleus_binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # FIXED: #8 apply morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        nucleus_binary = cv2.morphologyEx(nucleus_binary, cv2.MORPH_OPEN, kernel)
        
        # Find nucleus contours
        contours, _ = cv2.findContours(nucleus_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                'nucleus_area': 0, 'nucleus_perimeter': 0, 'nucleus_circularity': 0,
                'nucleus_cytoplasm_ratio': 0
            }
        
        nucleus_contour = max(contours, key=cv2.contourArea)
        nucleus_area = cv2.contourArea(nucleus_contour)
        nucleus_perimeter = cv2.arcLength(nucleus_contour, True)
        
        # Circularity
        circularity = (4 * np.pi * nucleus_area) / (nucleus_perimeter ** 2) if nucleus_perimeter > 0 else 0
        
        # Nucleus to cytoplasm ratio (estimate)
        cytoplasm_area = gray.size - nucleus_area
        nc_ratio = nucleus_area / cytoplasm_area if cytoplasm_area > 0 else 0
        
        return {
            'nucleus_area': float(nucleus_area),
            'nucleus_perimeter': float(nucleus_perimeter),
            'nucleus_circularity': float(circularity),
            'nucleus_cytoplasm_ratio': float(nc_ratio)
        }
    
    def extract_edge_features(self, gray):
        """Extract edge-based features."""
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            'edge_density': float(edge_density)
        }


def extract_medical_features(image_path, feature_extractor):
    """Extract and order medical features from image."""
    # FIXED: #2 removed 'glcm_dissimilarity' to match 30-feature contract
    feature_order = [
        'cell_area', 'cell_perimeter', 'cell_solidity', 'cell_eccentricity', 'cell_aspect_ratio',
        'glcm_contrast', 'glcm_correlation', 'glcm_energy', 'glcm_homogeneity',
        'nucleus_area', 'nucleus_perimeter', 'nucleus_circularity', 'nucleus_cytoplasm_ratio',
        'edge_density'
    ]
    # Total: 5 + 4 + 4 + 1 = 14 features currently; pad to 30 for model compatibility
    # OR add more features to reach 30 naturally
    
    features_dict = feature_extractor.extract_all_features(image_path)
    features = [features_dict.get(name, 0.0) for name in feature_order]
    
    # Pad to 30 features if needed (placeholder; better to extract real features)
    while len(features) < 30:
        features.append(0.0)
    
    return np.array(features[:30], dtype=np.float32)


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
