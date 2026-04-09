import torch
from diffusers import StableDiffusionPipeline
import os
import shutil
from pathlib import Path

class SyntheticDataGenerator:
    """
    GenAI model for generating synthetic cervical cancer images
    Uses Stable Diffusion for data augmentation
    
    Note: Safety checker is disabled for legitimate medical research purposes.
    """
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Stable Diffusion model on {self.device}...")
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipe = self.pipe.to(self.device)
        
    def generate_images(self, stage, num_images=10, output_dir="./data/synthetic", auto_merge=True):
        """
        Generate synthetic images for a specific cancer stage
        
        Args:
            stage: Cancer stage (Normal, CIN1, CIN2, CIN3, Cancer)
            num_images: Number of images to generate
            output_dir: Directory to save generated images
            auto_merge: Automatically copy to training folder (default: True)
        """
        os.makedirs(output_dir, exist_ok=True)
        stage_dir = os.path.join(output_dir, stage)
        os.makedirs(stage_dir, exist_ok=True)
        
        # Check existing images to avoid regeneration - find highest numbered file
        existing_files = [f for f in os.listdir(stage_dir) if f.startswith(f"{stage}_") and f.endswith('.png')]
        
        if existing_files:
            # Extract numbers from filenames like "CIN2_1.png", "CIN2_2.png"
            import re
            numbers = []
            for f in existing_files:
                match = re.search(rf'{stage}_(\d+)\.png', f)
                if match:
                    numbers.append(int(match.group(1)))
            
            existing_count = max(numbers) if numbers else 0
            print(f"Found {len(existing_files)} existing images for {stage} (up to {stage}_{existing_count}.png)")
            print(f"Will start from {stage}_{existing_count + 1}.png...")
        else:
            existing_count = 0
            print(f"No existing images found for {stage}")
        
        start_idx = existing_count
        
        # Define prompts for each stage with more abstract, technical descriptions
        prompts = {
            "Normal": "histopathology microscopy image, pink epithelial cells, normal tissue structure, medical slide, scientific photograph, clinical diagnostic image",
            "CIN1": "histopathology microscopy image, epithelial tissue with mild cellular changes, dysplastic cells, medical slide, scientific photograph, diagnostic pathology",
            "CIN2": "histopathology microscopy image, epithelial tissue with moderate cellular abnormalities, dysplastic tissue, medical slide, scientific photograph, diagnostic pathology",
            "CIN3": "histopathology microscopy image, epithelial tissue with severe cellular changes, high-grade dysplasia, medical slide, scientific photograph, diagnostic pathology",
            "Cancer": "histopathology microscopy image, malignant epithelial cells, invasive carcinoma tissue, medical slide, scientific photograph, diagnostic pathology"
        }
        
        # Negative prompt to avoid unwanted content
        negative_prompt = "blurry, low quality, cartoon, illustration, text, watermark, signature, person, face, body"
        
        prompt = prompts.get(stage, prompts["Normal"])
        
        images_to_generate = num_images - existing_count
        if images_to_generate <= 0:
            print(f"Already have {existing_count} images for {stage}, target was {num_images}. Skipping generation.")
            return []
        
        print(f"Generating {images_to_generate} NEW images for {stage}...")
        new_image_paths = []
        
        for i in range(images_to_generate):
            image = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
            
            image_path = os.path.join(stage_dir, f"{stage}_{start_idx + i + 1}.png")
            image.save(image_path)
            new_image_paths.append(image_path)
            print(f"Saved: {image_path}")
        
        print(f"Generated {images_to_generate} new images for {stage} (Total: {start_idx + images_to_generate})")
        
        # Auto-merge to training folder if enabled
        if auto_merge and new_image_paths:
            self._merge_to_training(stage, new_image_paths)
        
        return new_image_paths
        
    def _merge_to_training(self, stage, image_paths):
        """Copy synthetic images to training folder with 'synthetic_' prefix"""
        # Get absolute path to project root data/train folder
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent.parent
        train_dir = project_root / "data" / "train"
        train_stage_dir = train_dir / stage
        
        if not train_stage_dir.exists():
            print(f"⚠️  Training folder not found: {train_stage_dir}")
            return
        
        copied_count = 0
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            new_filename = f"synthetic_{filename}"
            dest_path = train_stage_dir / new_filename
            
            shutil.copy2(img_path, str(dest_path))
            copied_count += 1
        
        print(f"✓ Auto-merged: Copied {copied_count} images to {train_stage_dir}")
    
    def generate_all_stages(self, num_images_per_stage=10, output_dir="./data/synthetic", auto_merge=True):
        """Generate synthetic images for all cancer stages"""
        stages = ["Normal", "CIN1", "CIN2", "CIN3", "Cancer"]
        total_generated = 0
        
        for stage in stages:
            new_images = self.generate_images(stage, num_images_per_stage, output_dir, auto_merge)
            total_generated += len(new_images)
        
        print(f"\n{'='*60}")
        print(f"✅ Completed! Generated {total_generated} NEW images total")
        if auto_merge:
            print(f"✅ All new images automatically copied to training folders")
        print(f"{'='*60}")
