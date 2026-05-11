import argparse
import os
from pathlib import Path
from models.genai_model import SyntheticDataGenerator

def main():
    # Get absolute path to project root (parent of backend directory)
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    default_output_dir = project_root / "data" / "synthetic"
    
    parser = argparse.ArgumentParser(description='Generate Synthetic Cervical Cancer Images')
    parser.add_argument('--num-images', type=int, default=100,
                        help='Number of images to generate per stage')
    parser.add_argument('--output-dir', type=str, default=str(default_output_dir),
                        help='Output directory for synthetic images')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['Normal', 'CIN1', 'CIN2', 'CIN3', 'Cancer', 'all'],
                        help='Cancer stage to generate images for')
    parser.add_argument('--model', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='Stable Diffusion model to use')
    parser.add_argument('--no-auto-merge', action='store_true',
                        help='Disable automatic merging to training folder')
    
    args = parser.parse_args()
    auto_merge = not args.no_auto_merge
    
    print("=" * 60)
    print("Cervical Cancer Synthetic Data Generator")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Images per stage: {args.num_images}")
    print(f"Stage: {args.stage}")
    print(f"Auto-merge to training: {'Yes' if auto_merge else 'No'}")
    print("=" * 60)
    print("Note: Will skip existing images and generate only new ones")
    print("=" * 60)
    
    # Initialize generator
    generator = SyntheticDataGenerator(model_name=args.model)
    
    # Generate images
    if args.stage == 'all':
        generator.generate_all_stages(
            num_images_per_stage=args.num_images,
            output_dir=args.output_dir,
            auto_merge=auto_merge
        )
    else:
        new_images = generator.generate_images(
            stage=args.stage,
            num_images=args.num_images,
            output_dir=args.output_dir,
            auto_merge=auto_merge
        )
        if auto_merge and new_images:
            print(f"\n✅ {len(new_images)} new images auto-merged to training folder")
    
    print("\n✓ Generation complete!")
    print(f"Images saved to: {args.output_dir}")
    if auto_merge:
        print(f"✅ New images automatically copied to data/train/")

if __name__ == "__main__":
    main()
