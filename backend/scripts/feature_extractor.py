"""
Traditional ML Feature Extraction for Cervical Cancer Classification
Extracts hand-crafted features: morphology, texture (GLCM), color, nucleus, edge.

FIXES APPLIED:
  - GLCM: do NOT slice glcm to 2D before graycoprops — pass full 4D array
  - extract_medical_features: accepts PIL Image (not image_path + extractor)
  - extract_color_features: restored (was dropped in Copilot rewrite)
  - Feature order: 26 real features + 4 zero-pads = 30 total (matches model)
  - CellFeatureExtractor.extract_all_features: accepts numpy array, not path
  - Morphological opening on nucleus binary (fix #8)
"""

import cv2
import numpy as np

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import graycomatrix, graycoprops  # older skimage

from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor class
# ─────────────────────────────────────────────────────────────────────────────

class CellFeatureExtractor:
    """Extract morphological, texture, color, nucleus and edge features
    from a single cervical cell image (numpy RGB array, uint8)."""

    # ── Morphology ────────────────────────────────────────────────────────────
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
                "cell_area": 0.0,
                "cell_perimeter": 0.0,
                "cell_solidity": 0.0,
                "cell_eccentricity": 0.0,
                "cell_aspect_ratio": 0.0,
            }

        cell_contour = max(contours, key=cv2.contourArea)
        cell_area = cv2.contourArea(cell_contour)
        cell_perimeter = cv2.arcLength(cell_contour, True)

        # Ellipse fit → eccentricity + aspect ratio
        if len(cell_contour) >= 5:
            _, (major, minor), _ = cv2.fitEllipse(cell_contour)
            eccentricity = (
                float(np.sqrt(1.0 - (minor / major) ** 2)) if major > 0 else 0.0
            )
            aspect_ratio = float(major / minor) if minor > 0 else 0.0
        else:
            eccentricity = 0.0
            aspect_ratio = 1.0

        # Solidity
        hull = cv2.convexHull(cell_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(cell_area / hull_area) if hull_area > 0 else 0.0

        return {
            "cell_area": float(cell_area),
            "cell_perimeter": float(cell_perimeter),
            "cell_solidity": solidity,
            "cell_eccentricity": eccentricity,
            "cell_aspect_ratio": aspect_ratio,
        }

    # ── Texture (GLCM) ────────────────────────────────────────────────────────
    def extract_texture_features(self, gray):
        """4 features: glcm_contrast, glcm_correlation, glcm_energy,
        glcm_homogeneity.

        CRITICAL FIX: graycoprops() requires the FULL 4-D GLCM array
        (levels × levels × n_distances × n_angles).
        Do NOT slice it down to 2-D before calling graycoprops — that
        raises 'The parameter P must be a 4-dimensional array' on every call.
        """
        try:
            if gray.ndim != 2:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.uint8)

            # glcm.shape == (256, 256, 1, 1) — keep it 4-D for graycoprops
            glcm = graycomatrix(
                gray,
                distances=[1],
                angles=[0],
                levels=256,
                symmetric=True,
                normed=True,
            )

            if glcm.size == 0 or np.isnan(glcm).any():
                raise ValueError("Invalid GLCM matrix")

            # Index [0, 0] picks distance-0 / angle-0 scalar after graycoprops
            return {
                "glcm_contrast":    float(graycoprops(glcm, "contrast")[0, 0]),
                "glcm_correlation": float(graycoprops(glcm, "correlation")[0, 0]),
                "glcm_energy":      float(graycoprops(glcm, "energy")[0, 0]),
                "glcm_homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
            }

        except Exception:
            return {
                "glcm_contrast": 0.0,
                "glcm_correlation": 0.0,
                "glcm_energy": 0.0,
                "glcm_homogeneity": 0.0,
            }

    # ── Color (RGB + HSV) ─────────────────────────────────────────────────────
    def extract_color_features(self, image_rgb):
        """12 features: {red,green,blue}_{mean,std} + {hue,sat,val}_{mean,std}.

        Accepts uint8 RGB numpy array.
        """
        features = {}
        for i, ch in enumerate(["red", "green", "blue"]):
            features[f"{ch}_mean"] = float(np.mean(image_rgb[:, :, i]))
            features[f"{ch}_std"]  = float(np.std(image_rgb[:, :, i]))

        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        for i, ch in enumerate(["hue", "saturation", "value"]):
            features[f"{ch}_mean"] = float(np.mean(hsv[:, :, i]))
            features[f"{ch}_std"]  = float(np.std(hsv[:, :, i]))

        return features

    # ── Nucleus ───────────────────────────────────────────────────────────────
    def extract_nucleus_features(self, gray):
        """4 features: nucleus_area, nucleus_perimeter, nucleus_circularity,
        nucleus_cytoplasm_ratio.

        FIX #8: morphological opening removes small noise blobs before
        contour detection, stabilising nucleus_area / NC-ratio.
        """
        inverted = cv2.bitwise_not(gray)
        _, nucleus_binary = cv2.threshold(
            inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # FIX #8 — morphological opening (5×5 ellipse kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        nucleus_binary = cv2.morphologyEx(nucleus_binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            nucleus_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {
                "nucleus_area": 0.0,
                "nucleus_perimeter": 0.0,
                "nucleus_circularity": 0.0,
                "nucleus_cytoplasm_ratio": 0.0,
            }

        nucleus_contour = max(contours, key=cv2.contourArea)
        nucleus_area    = cv2.contourArea(nucleus_contour)
        nucleus_perim   = cv2.arcLength(nucleus_contour, True)

        circularity = (
            float((4.0 * np.pi * nucleus_area) / (nucleus_perim ** 2))
            if nucleus_perim > 0
            else 0.0
        )

        total_area     = float(gray.shape[0] * gray.shape[1])
        cytoplasm_area = max(total_area - nucleus_area, 1.0)
        nc_ratio       = float(nucleus_area / cytoplasm_area)

        return {
            "nucleus_area":             float(nucleus_area),
            "nucleus_perimeter":        float(nucleus_perim),
            "nucleus_circularity":      circularity,
            "nucleus_cytoplasm_ratio":  nc_ratio,
        }

    # ── Edge ──────────────────────────────────────────────────────────────────
    def extract_edge_features(self, gray):
        """1 feature: edge_density."""
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)
        return {"edge_density": edge_density}

    # ── Convenience: all features from numpy RGB array ────────────────────────
    def extract_all_features(self, image_rgb):
        """
        Extract all features from a uint8 numpy RGB array.
        Returns a flat dict with all feature keys.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        features = {}
        features.update(self.extract_morphology_features(gray))
        features.update(self.extract_texture_features(gray))
        features.update(self.extract_color_features(image_rgb))
        features.update(self.extract_nucleus_features(gray))
        features.update(self.extract_edge_features(gray))
        return features


# ─────────────────────────────────────────────────────────────────────────────
# Public API used by train_hybrid.py and inference code
# ─────────────────────────────────────────────────────────────────────────────

# Canonical feature order — 26 real features + 4 reserved zeros = 30 total.
# Must match NUM_TRADITIONAL_FEATURES = 30 in train_hybrid.py.
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


def extract_medical_features(image):
    """
    Extract 30 medical features from a single image.

    Args:
        image: PIL Image  OR  uint8 numpy array (H, W, 3) in RGB order.

    Returns:
        List[float] of length 30, ready to be fed to StandardScaler / model.

    FIXES vs previous version:
        - Accepts PIL Image directly (train_hybrid.py passes PIL images).
        - No 'image_path' or 'feature_extractor' arguments needed.
        - GLCM uses full 4-D array — no more 'P must be 4-dimensional' errors.
        - Color features restored.
        - Pads to exactly 30 features.
    """
    extractor = CellFeatureExtractor()

    # Accept PIL Image
    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"), dtype=np.uint8)

    # Ensure uint8 RGB numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Resize to canonical 224×224 for consistent feature extraction
    image_resized = cv2.resize(image, (224, 224))

    features_dict = extractor.extract_all_features(image_resized)

    # Build ordered list; padding keys default to 0.0
    features = [features_dict.get(key, 0.0) for key in _FEATURE_ORDER]
    return features  # length == 30


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level batch extraction (unchanged from original, kept for CLI use)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_from_dataset(data_dir, output_csv="features.csv"):
    """Extract features from all images in a dataset folder and save to CSV."""
    import pandas as pd
    from tqdm import tqdm

    extractor = CellFeatureExtractor()
    data_dir  = Path(data_dir)

    all_features = []
    classes      = ["Normal", "CIN1", "CIN2", "CIN3", "Cancer"]

    print("Extracting hand-crafted features from images...")

    for class_name in classes:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found")
            continue

        image_files = (
            list(class_dir.glob("*.png"))
            + list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
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
                print(f"Error processing {img_path.name}: {e}")

    df = pd.DataFrame(all_features)
    output_path = data_dir.parent / output_csv
    df.to_csv(output_path, index=False)

    print(f"\nFeatures saved to: {output_path}")
    print(f"Total samples   : {len(df)}")
    print(f"Total features  : {len(df.columns) - 2}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract traditional ML features from cervical images"
    )
    parser.add_argument(
        "--data-dir", type=str, default="../data/train",
        help="Directory containing class sub-folders of images"
    )
    parser.add_argument(
        "--output", type=str, default="train_features.csv",
        help="Output CSV filename"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a quick smoke test with a dummy image instead of full extraction"
    )
    args = parser.parse_args()

    if args.smoke_test:
        print("Running smoke test...")
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        feats = extract_medical_features(dummy_img)
        assert len(feats) == 30, f"Expected 30 features, got {len(feats)}"
        print(f"  extract_medical_features → {len(feats)} features  ✓")
        print(f"  First 5 values: {[round(f, 4) for f in feats[:5]]}")
        print("Smoke test passed!")
    else:
        extract_features_from_dataset(args.data_dir, args.output)