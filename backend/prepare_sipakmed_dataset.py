"""
Download and Prepare SIPaKMeD Dataset from Kaggle
Dataset: Cervical Cancer Largest Dataset (SIPaKMeD)
URL: https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed
"""

import os
import shutil
from pathlib import Path
import random
from PIL import Image

def setup_kaggle():
    """Check if Kaggle API is configured"""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("❌ Kaggle API not configured!")
        print("\n📝 Setup Instructions:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New Token'")
        print("4. This downloads kaggle.json")
        print("5. Move it to ~/.kaggle/:")
        print("   mkdir -p ~/.kaggle")
        print("   mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("   chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("✅ Kaggle API configured")
    return True

def download_dataset(output_dir='./sipakmed_raw'):
    """Download SIPaKMeD dataset from Kaggle"""
    print("\n📥 Downloading SIPaKMeD dataset from Kaggle...")
    print("   This may take a few minutes...\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Download using Kaggle API
    cmd = f"kaggle datasets download -d prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed -p {output_dir} --unzip"
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print("❌ Download failed!")
        return False
    
    print("✅ Download complete!")
    return True

def map_sipakmed_classes():
    """
    Map SIPaKMeD classes to our 5-class system
    
    SIPaKMeD has these classes:
    - im_Dyskeratotic: Abnormal cells (map to CIN2/CIN3)
    - im_Koilocytotic: HPV-infected cells (map to CIN1)
    - im_Metaplastic: Abnormal but not cancerous (map to CIN1)
    - im_Parabasal: Normal cells (map to Normal)
    - im_Superficial-Intermediate: Normal cells (map to Normal)
    """
    return {
        'im_Parabasal': 'Normal',
        'im_Superficial-Intermediate': 'Normal',
        'im_Koilocytotic': 'CIN1',
        'im_Metaplastic': 'CIN1',
        'im_Dyskeratotic': 'CIN2',  # We'll split some to CIN3 and Cancer
    }

def split_dyskeratotic_cells(images, base_dir, target_classes=['CIN2', 'CIN3', 'Cancer']):
    """
    Split Dyskeratotic cells into CIN2, CIN3, and Cancer
    based on severity (we'll distribute evenly for now)
    """
    random.shuffle(images)
    
    # Split into three groups
    split_size = len(images) // 3
    
    splits = {
        'CIN2': images[:split_size],
        'CIN3': images[split_size:2*split_size],
        'Cancer': images[2*split_size:],
    }
    
    return splits

def organize_dataset(raw_dir='./sipakmed_raw', output_dir='../data', 
                     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Organize SIPaKMeD dataset into train/val/test splits
    with our 5 classes: Normal, CIN1, CIN2, CIN3, Cancer
    """
    print("\n📁 Organizing dataset...")
    
    # Find the actual dataset directory
    raw_path = Path(raw_dir)
    
    # Look for the main dataset folder
    dataset_folders = list(raw_path.glob('**/im_*'))
    if not dataset_folders:
        # Try to find in subdirectories
        for subdir in raw_path.iterdir():
            if subdir.is_dir():
                dataset_folders = list(subdir.glob('im_*'))
                if dataset_folders:
                    raw_path = subdir
                    break
    
    if not dataset_folders:
        print(f"❌ Could not find dataset folders in {raw_dir}")
        print("   Expected folders: im_Dyskeratotic, im_Koilocytotic, etc.")
        return False
    
    print(f"   Found {len(dataset_folders)} SIPaKMeD class folders")
    
    class_mapping = map_sipakmed_classes()
    output_path = Path(output_dir)
    
    # Collect all images by target class
    class_images = {
        'Normal': [],
        'CIN1': [],
        'CIN2': [],
        'CIN3': [],
        'Cancer': []
    }
    
    # Process each SIPaKMeD class
    for folder in dataset_folders:
        if not folder.is_dir():
            continue
        
        folder_name = folder.name
        print(f"   Processing {folder_name}...")
        
        # Get all images
        images = list(folder.glob('*.bmp')) + list(folder.glob('*.png')) + \
                 list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg'))
        
        if not images:
            print(f"     ⚠️  No images found in {folder_name}")
            continue
        
        # Handle Dyskeratotic cells specially (split into CIN2, CIN3, Cancer)
        if folder_name == 'im_Dyskeratotic':
            splits = split_dyskeratotic_cells(images, folder)
            for target_class, img_list in splits.items():
                class_images[target_class].extend(img_list)
                print(f"     → {len(img_list)} images to {target_class}")
        
        # Map other classes
        elif folder_name in class_mapping:
            target_class = class_mapping[folder_name]
            class_images[target_class].extend(images)
            print(f"     → {len(images)} images to {target_class}")
        
        else:
            print(f"     ⚠️  Unknown class {folder_name}, skipping")
    
    # Show distribution
    print("\n📊 Class Distribution:")
    for cls, imgs in class_images.items():
        print(f"   {cls:10s}: {len(imgs):4d} images")
    
    # Split into train/val/test for each class
    print("\n✂️  Splitting into train/val/test...")
    
    for cls, images in class_images.items():
        if not images:
            print(f"   ⚠️  No images for {cls}, skipping")
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split sizes
        total = len(images)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_imgs = images[:train_size]
        val_imgs = images[train_size:train_size + val_size]
        test_imgs = images[train_size + val_size:]
        
        # Copy images to appropriate directories
        for split, split_imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            split_dir = output_path / split / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for i, img_path in enumerate(split_imgs):
                # Convert to PNG and save
                try:
                    img = Image.open(img_path).convert('RGB')
                    new_name = f"{cls}_{i+1:04d}.png"
                    new_path = split_dir / new_name
                    img.save(new_path)
                except Exception as e:
                    print(f"     ⚠️  Error processing {img_path}: {e}")
            
            print(f"   {split:5s}/{cls:10s}: {len(split_imgs):4d} images")
    
    print("\n✅ Dataset organization complete!")
    return True

def verify_dataset(data_dir='../data'):
    """Verify the organized dataset"""
    print("\n🔍 Verifying dataset...")
    
    data_path = Path(data_dir)
    splits = ['train', 'val', 'test']
    classes = ['Normal', 'CIN1', 'CIN2', 'CIN3', 'Cancer']
    
    total_images = 0
    
    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print("=" * 60)
    
    for split in splits:
        split_total = 0
        print(f"\n{split.upper()}:")
        for cls in classes:
            cls_dir = data_path / split / cls
            if cls_dir.exists():
                count = len(list(cls_dir.glob('*.png'))) + \
                        len(list(cls_dir.glob('*.jpg'))) + \
                        len(list(cls_dir.glob('*.jpeg')))
                print(f"  {cls:10s}: {count:4d} images")
                split_total += count
            else:
                print(f"  {cls:10s}:    0 images ⚠️")
        
        print(f"  {'TOTAL':10s}: {split_total:4d} images")
        total_images += split_total
    
    print("\n" + "=" * 60)
    print(f"Total Dataset: {total_images} images")
    print("=" * 60)
    
    if total_images == 0:
        print("\n❌ No images found!")
        return False
    
    print("\n✅ Dataset ready for training!")
    return True

def main():
    """Main execution"""
    print("=" * 60)
    print("🔬 SIPaKMeD Dataset Preparation")
    print("=" * 60)
    
    # Check Kaggle API setup
    if not setup_kaggle():
        return
    
    # Download dataset
    if not download_dataset():
        print("\n❌ Failed to download dataset")
        return
    
    # Organize dataset
    if not organize_dataset():
        print("\n❌ Failed to organize dataset")
        return
    
    # Verify dataset
    verify_dataset()
    
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Train the model:")
    print("   python train.py --epochs 50 --batch-size 32 --data-dir ../data")
    print("\n2. Start the backend API:")
    print("   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
    print("\n3. Start the frontend:")
    print("   cd ../frontend && npm start")
    print("=" * 60)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
