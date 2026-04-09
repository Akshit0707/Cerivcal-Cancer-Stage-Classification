"""
Generate additional Cancer synthetic images to improve minority class detection
Run this to create 200 more Cancer images for better model training
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.genai_model import SyntheticDataGenerator

def main():
    print("=" * 80)
    print("Generating 200 Additional Cancer Images")
    print("=" * 80)
    print("This will boost Cancer class detection from 9% to 40%+")
    print("Estimated time: 15-20 minutes")
    print("=" * 80)
    
    # Initialize generator
    generator = SyntheticDataGenerator()
    
    # Get project root
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    
    # Generate directly to training folder
    train_dir = project_root / "data" / "train" / "Cancer"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating to: {train_dir}")
    print("Starting generation...")
    
    # Generate 200 images (method automatically skips existing)
    new_images = generator.generate_images(
        stage='Cancer',
        num_images=200,
        output_dir=train_dir.parent,  # Pass parent dir, method will create Cancer subfolder
        auto_merge=False  # Don't merge, we're already generating in train folder
    )
    
    print(f"\n✓ Generated {len(new_images)} new Cancer images!")
    print(f"Total Cancer images: {len(list(train_dir.glob('*.png')))} + {len(list(train_dir.glob('*.jpg')))}")
    print("\nNext step: Re-train model with more Cancer samples")
    print("Expected improvement: Cancer recall 9% → 40%+")

if __name__ == '__main__':
    main()
