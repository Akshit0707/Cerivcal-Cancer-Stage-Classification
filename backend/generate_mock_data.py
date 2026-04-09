"""
Generate Mock Data for Testing
This creates placeholder images to test the pipeline WITHOUT real medical data
"""
import os
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np

def create_mock_image(stage, image_num, size=(224, 224)):
    """Create a mock cervical cancer image with text label"""
    
    # Create base image with random medical-like texture
    img = Image.new('RGB', size)
    pixels = img.load()
    
    # Generate textured background
    for i in range(size[0]):
        for j in range(size[1]):
            # Create tissue-like colors (pinkish/reddish hues)
            base_color = random.randint(180, 255)
            r = min(255, base_color + random.randint(-30, 30))
            g = min(255, int(base_color * 0.7) + random.randint(-20, 20))
            b = min(255, int(base_color * 0.7) + random.randint(-20, 20))
            pixels[i, j] = (r, g, b)
    
    # Add some circular patterns (simulating tissue structures)
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(3, 8)):
        x = random.randint(20, size[0]-20)
        y = random.randint(20, size[1]-20)
        r = random.randint(10, 30)
        color = (
            random.randint(150, 200),
            random.randint(100, 150),
            random.randint(100, 150)
        )
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=(100, 50, 50))
    
    # Add label text
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    text = f"MOCK {stage} #{image_num}"
    draw.text((10, 10), text, fill=(50, 50, 50), font=font)
    
    # Add warning watermark
    try:
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font_small = ImageFont.load_default()
    
    draw.text((10, size[1]-25), "SYNTHETIC TEST DATA", fill=(255, 0, 0), font=font_small)
    
    return img

def generate_mock_dataset(base_dir='../data', images_per_class=20):
    """Generate mock dataset for testing"""
    
    stages = ['Normal', 'CIN1', 'CIN2', 'CIN3', 'Cancer']
    splits = ['train', 'val', 'test']
    
    # Distribution: 70% train, 15% val, 15% test
    split_counts = {
        'train': int(images_per_class * 0.7),
        'val': int(images_per_class * 0.15),
        'test': images_per_class - int(images_per_class * 0.7) - int(images_per_class * 0.15)
    }
    
    print("=" * 60)
    print("🧪 Generating MOCK Dataset for Testing")
    print("=" * 60)
    print("\n⚠️  WARNING: This is NOT real medical data!")
    print("   Use only for testing the pipeline.\n")
    
    total_images = 0
    
    for split in splits:
        for stage in stages:
            dir_path = os.path.join(base_dir, split, stage)
            os.makedirs(dir_path, exist_ok=True)
            
            num_images = split_counts[split]
            
            print(f"Creating {num_images} images for {split}/{stage}...", end=' ')
            
            for i in range(num_images):
                img = create_mock_image(stage, i+1)
                img_path = os.path.join(dir_path, f"{stage}_mock_{i+1:03d}.png")
                img.save(img_path)
                total_images += 1
            
            print("✓")
    
    print("\n" + "=" * 60)
    print(f"✅ Created {total_images} mock images")
    print("=" * 60)
    print("\nDataset Distribution:")
    for split in splits:
        count = split_counts[split] * len(stages)
        print(f"  {split:6s}: {count} images ({split_counts[split]} per class)")
    
    print("\n📊 Classes: Normal, CIN1, CIN2, CIN3, Cancer")
    print("\n⚠️  IMPORTANT:")
    print("   - These are SYNTHETIC placeholder images")
    print("   - Use ONLY for testing the pipeline")
    print("   - Replace with real medical data for actual use")
    print("   - Model trained on this won't work for real diagnosis")
    print("\n🔍 To use real data:")
    print("   - Get proper medical image dataset")
    print("   - Replace mock images in data/train, data/val, data/test")
    print("   - Ensure proper medical licensing and ethics approval")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    
    # Get number of images per class from command line or use default
    images_per_class = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    
    print(f"\nGenerating {images_per_class} images per class...")
    generate_mock_dataset(images_per_class=images_per_class)
    
    print("\n✅ Mock dataset generation complete!")
    print("\nNext steps:")
    print("  1. Train the model: python train.py --epochs 10 --batch-size 16")
    print("  2. Test the system to verify it works")
    print("  3. Replace with REAL medical data before actual use")
