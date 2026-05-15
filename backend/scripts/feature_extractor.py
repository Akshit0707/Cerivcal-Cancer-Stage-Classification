"""
Traditional ML Feature Extraction for Cervical Cancer Classification
Extracts hand-crafted features: morphology, texture (GLCM), color, nucleus, edge.

Classes: Normal, CIN1, CIN2, CIN3, Cancer
"""

import cv2
import numpy as np

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import graycomatrix, graycoprops

from pathlib import Path


# ---------------------------------------------------------------------------
# Feature extractor class
# ---------------------------------------------------------------------------

class CellFeatureExtractor:
    """Extract morphological, texture, color, nucleus and edge features
    from a single cervical cell image (numpy RGB array, uint8)."""

    def extract_morphology_features(self, gray):
        """5 features: cell_area, cell_perimeter, cell_solidity,
        cell_eccentricity, cell_aspect_ratio."""
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {
                "cell_area": 0.0, "cell_perimeter": 0.0,
                "cell_solidity": 0.0, "cell_eccentricity": 0.0,
                "cell_aspect_ratio": 0.0,
            }

        cell_contour = max(contours, key=cv2.contourArea)
        cell_area    = cv2.contourArea(cell_contour)
        cell_perim   = cv2.arcLength(cell_contour, True)

        if len(cell_contour) >= 5:
            _, (major, minor), _ = cv2.fitEllipse(cell_contour)
            eccentricity = float(np.sqrt(1.0 - (minor / major) ** 2)) if major > 0 else 0.0
            aspect_ratio = float(major / minor) if minor > 0 else 0.0
        else:
            eccentricity = 0.0
            aspect_ratio = 1.0

        hull      = cv2.convexHull(cell_contour)
        hull_area = cv2.contourArea(hull)
        solidity  = float(cell_area / hull_area) if hull_area > 0 else 0.0

        return {
            "cell_area": float(cell_area), "cell_perimeter": float(cell_perim),
            "cell_solidity": solidity, "cell_eccentricity": eccentricity,
            "cell_aspect_ratio": aspect_ratio,
        }

    def extract_texture_features(self, gray):
        """4 features: glcm_contrast, glcm_correlation, glcm_energy, glcm_homogeneity."""
        try:
            if gray.ndim != 2:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.uint8)
            glcm = graycomatrix(gray, distances=[1], angles=[0],
                                levels=256, symmetric=True, normed=True)
            if glcm.size == 0 or np.isnan(glcm).any():
                raise ValueError("Invalid GLCM")
            return {
                "glcm_contrast":    float(graycoprops(glcm, "contrast")[0, 0]),
                "glcm_correlation": float(graycoprops(glcm, "correlation")[0, 0]),
                "glcm_energy":      float(graycoprops(glcm, "energy")[0, 0]),
                "glcm_homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
            }
        except Exception:
            return {"glcm_contrast": 0.0, "glcm_correlation": 0.0,
                    "glcm_energy": 0.0, "glcm_homogeneity": 0.0}

    def extract_color_features(self, image_rgb):
        """12 features: {red,green,blue}_{mean,std} + {hue,sat,val}_{mean,std}."""
        features = {}
        for i, ch in enumerate(["red", "green", "blue"]):
            features[f"{ch}_mean"] = float(np.mean(image_rgb[:, :, i]))
            features[f"{ch}_std"]  = float(np.std(image_rgb[:, :, i]))
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        for i, ch in enumerate(["hue", "saturation", "value"]):
            features[f"{ch}_mean"] = float(np.mean(hsv[:, :, i]))
            features[f"{ch}_std"]  = float(np.std(hsv[:, :, i]))
        return features

    def extract_nucleus_features(self, gray):
        """4 features: nucleus_area, nucleus_perimeter, nucleus_circularity,
        nucleus_cytoplasm_ratio."""
        inverted = cv2.bitwise_not(gray)
        _, nucleus_binary = cv2.threshold(
            inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        nucleus_binary = cv2.morphologyEx(nucleus_binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            nucleus_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return {"nucleus_area": 0.0, "nucleus_perimeter": 0.0,
                    "nucleus_circularity": 0.0, "nucleus_cytoplasm_ratio": 0.0}

        nc      = max(contours, key=cv2.contourArea)
        n_area  = cv2.contourArea(nc)
        n_perim = cv2.arcLength(nc, True)
        circ    = float((4.0 * np.pi * n_area) / (n_perim ** 2)) if n_perim > 0 else 0.0
        total   = float(gray.shape[0] * gray.shape[1])
        nc_ratio = float(n_area / max(total - n_area, 1.0))
        return {
            "nucleus_area": float(n_area), "nucleus_perimeter": float(n_perim),
            "nucleus_circularity": circ, "nucleus_cytoplasm_ratio": nc_ratio,
        }

    def extract_edge_features(self, gray):
        """1 feature: edge_density."""
        edges = cv2.Canny(gray, 50, 150)
        return {"edge_density": float(np.sum(edges > 0) / edges.size)}

    def extract_all_features(self, image_rgb):
        """Extract all features from a uint8 numpy RGB array."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        features = {}
        features.update(self.extract_morphology_features(gray))
        features.update(self.extract_texture_features(gray))
        features.update(self.extract_color_features(image_rgb))
        features.update(self.extract_nucleus_features(gray))
        features.update(self.extract_edge_features(gray))
        return features


# ---------------------------------------------------------------------------
# Canonical feature order — 26 real features + 4 zero-pads = 30 total
# ---------------------------------------------------------------------------
_FEATURE_ORDER = [
    # morphology (5)
    "cell_area", "cell_perimeter", "cell_solidity",
    "cell_eccentricity", "cell_aspect_ratio",
    # texture (4)
    "glcm_contrast", "glcm_correlation", "glcm_energy", "glcm_homogeneity",
    # color RGB (6)
    "red_mean", "green_mean", "blue_mean",
    "red_std",  "green_std",  "blue_std",
    # color HSV (6)
    "hue_mean", "saturation_mean", "value_mean",
    "hue_std",  "saturation_std",  "value_std",
    # nucleus (4)
    "nucleus_area", "nucleus_perimeter",
    "nucleus_circularity", "nucleus_cytoplasm_ratio",
    # edge (1)
    "edge_density",
    # reserved padding to reach 30 (4 zeros)
    "_pad0", "_pad1", "_pad2", "_pad3",
]

assert len(_FEATURE_ORDER) == 30, "Feature order must have exactly 30 entries"

# Class names used throughout the project
CLASS_NAMES = ['CIN1', 'CIN2', 'CIN3', 'Normal', 'Cancer']


def extract_medical_features(image):
    """
    Extract 30 medical features from a single image.

    Args:
        image: PIL Image OR uint8 numpy array (H, W, 3) RGB.

    Returns:
        List[float] of length 30.
    """
    extractor = CellFeatureExtractor()

    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"), dtype=np.uint8)

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    image_resized = cv2.resize(image, (224, 224))
    features_dict = extractor.extract_all_features(image_resized)
    return [features_dict.get(key, 0.0) for key in _FEATURE_ORDER]


# ---------------------------------------------------------------------------
# Dataset-level batch extraction
# ---------------------------------------------------------------------------

def extract_features_from_dataset(data_dir, output_csv="features.csv"):
    """Extract features from all images in a dataset folder and save to CSV.

    Expected folder structure:
        data_dir/
          Normal/
          CIN1/
          CIN2/
          CIN3/
          Cancer/
    """
    import pandas as pd
    from tqdm import tqdm

    extractor = CellFeatureExtractor()
    data_dir  = Path(data_dir)
    all_features = []

    print("Extracting hand-crafted features from images...")

    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found — skipping")
            continue

        image_files = (
            list(class_dir.glob("*.png")) +
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.jpeg"))
        )
        print(f"\nProcessing {class_name}: {len(image_files)} images")

        for img_path in tqdm(image_files, desc=f"  {class_name}"):
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                feat = extractor.extract_all_features(image)
                feat["class"]    = class_name
                feat["filename"] = img_path.name
                all_features.append(feat)
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")

    df = pd.DataFrame(all_features)
    output_path = data_dir.parent / output_csv
    df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to : {output_path}")
    print(f"Total samples     : {len(df)}")
    print(f"Total features    : {len(df.columns) - 2}")
    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/train")
    parser.add_argument("--output",   default="train_features.csv")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        print("Running smoke test...")
        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        feats = extract_medical_features(dummy)
        assert len(feats) == 30, f"Expected 30 features, got {len(feats)}"
        print(f"  extract_medical_features -> {len(feats)} features  OK")
        print(f"  Classes: {CLASS_NAMES}")
        print("Smoke test passed!")
    else:
        extract_features_from_dataset(args.data_dir, args.output)