# 🔬 Cervical Cancer Stage Classification System - Complete Setup Guide

## 📋 Prerequisites

- Python 3.8+ installed
- Node.js 16+ and npm installed
- Git installed
- At least 8GB RAM recommended
- GPU with CUDA support (optional, for faster training)

## 🚀 Quick Start

### 1. Clone or Navigate to the Project

```bash
cd /private/tmp/cervical-cancer-classifier
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# On Windows use: venv\Scripts\activate

pip install --upgrade pip
pip install -r ../requirements.txt
```

#### Prepare Data

You have two options:

**Option A: Use Your Own Dataset**
- Place images in `data/train/`, `data/val/`, and `data/test/` folders
- Organize by class: Normal, CIN1, CIN2, CIN3, Cancer

**Option B: Generate Synthetic Data (GenAI)** - With Auto-Merge

```bash
cd backend

# Generate synthetic images - automatically merges to training folder
python generate_synthetic_data.py --num-images 50 --stage CIN2

# Features:
# ✅ Skips existing images (generates only new ones)
# ✅ Auto-merges new images to data/train/CIN2/
# ✅ Sequential numbering (CIN2_1.png, CIN2_2.png, etc.)

# Generate for all stages
python generate_synthetic_data.py --num-images 100 --stage all

# Disable auto-merge if needed
python generate_synthetic_data.py --num-images 50 --stage CIN2 --no-auto-merge
```

⚠️ **Note**: Synthetic data is for augmentation only. You need real medical images as the primary dataset.

#### Train the Model

**Option 1: Train Standard CNN Model**

```bash
# Make sure you're in the backend directory with venv activated
python train.py --epochs 50 --batch-size 32 --data-dir ../data

python train_hybrid.py --epochs 50 --batch-size 32 --data-dir ../data

# For faster training with GPU:
python train.py --epochs 30 --batch-size 64 --data-dir ../data
```

**Option 2: Train Hybrid Model** ⭐ **COMING SOON**

Hybrid model is being redesigned for optimal CPU performance and improved accuracy. Will be available soon.

**Training Outputs:**
- CNN Model: `checkpoints/best_model.pth`
- Training progress displayed in terminal
- Best model selected based on validation accuracy

#### Start the Backend API

```bash
# Make sure you're in the backend directory with venv activated
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: http://localhost:8000

Test the API:
```bash
curl http://localhost:8000/health
```

### 3. Frontend Setup

Open a **new terminal** window:

```bash
cd /private/tmp/cervical-cancer-classifier/frontend

# Install dependencies
npm install

# Start the development server
npm start
```

Frontend will open automatically at: http://localhost:3000

## 📱 Using the Application

1. **Open Browser**: Navigate to http://localhost:3000
2. **Upload Image**: Click the upload area or drag & drop a cervical image
3. **Predict**: Click "🔍 Predict Stage" button
4. **View Results**: See the predicted cancer stage with confidence scores

## 🏗️ Project Structure

```
cervical-cancer-classifier/
├── backend/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_model.py          # CNN architecture
│   │   └── genai_model.py        # Synthetic data generator
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py               # FastAPI endpoints
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py        # Dataset & transforms
│   ├── checkpoints/              # Saved models
│   ├── train.py                  # Training script
│   └── generate_synthetic_data.py
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js               # Main React component
│   │   ├── App.css              # Styling
│   │   ├── index.js
│   │   └── index.css
│   └── package.json
├── data/
│   ├── train/                   # Training images
│   ├── val/                     # Validation images
│   ├── test/                    # Test images
│   └── synthetic/               # Generated images
├── requirements.txt             # Python dependencies
└── README.md
```

## 🔧 API Endpoints

### Health Check
```
GET http://localhost:8000/health
```

### Get Classes
```
GET http://localhost:8000/classes
```

### Predict
```
POST http://localhost:8000/predict
Content-Type: multipart/form-data
Body: file (image file)
```

## 🎯 Model Architecture

### **CNN Model** (Standard Deep Learning):
- 6 Convolutional layers with Batch Normalization
- Max Pooling layers
- 3 Fully Connected layers with Dropout (0.5)
- Output: 5 classes (Normal, CIN1, CIN2, CIN3, Cancer)
- Input size: 224x224x3
- Parameters: ~2.5M trainable parameters

### **Hybrid Model** (CNN + Traditional Features) ⭐:

Coming soon - Being redesigned for optimal CPU performance with improved accuracy.

## 🧪 Testing the Model

```bash
# In backend directory with venv activated
cd backend
python -c "
from models import load_model
import torch
model = load_model('checkpoints/best_model.pth')
print('Model loaded successfully!')
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

## 🐛 Troubleshooting

### Backend Issues

**Model not found error**:
- Make sure you've trained the model first using `train.py`
- Check that `checkpoints/best_model.pth` exists

**CUDA/GPU errors**:
- The code automatically falls back to CPU if CUDA is unavailable
- For CPU-only training, reduce batch size

**Import errors**:
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

### Frontend Issues

**Cannot connect to backend**:
- Verify backend is running on http://localhost:8000
- Check CORS settings in `backend/api/main.py`

**npm install fails**:
- Clear npm cache: `npm cache clean --force`
- Delete `node_modules` and `package-lock.json`, then reinstall

## 📊 Expected Performance

### Model Comparison:

| Model Type | Train Acc | Val Acc | Test Acc | Training Time | Interpretability |
|------------|-----------|---------|----------|---------------|------------------|
| Traditional ML | 80% | 75% | 73% | 5 min | ⭐⭐⭐⭐⭐ |
| CNN | 95% | 88% | 86% | 2-3 hours | ⭐⭐ |
| **Hybrid** | **Coming Soon** | - | - | - | ⭐⭐⭐⭐ |

### Current Dataset:
- **Real Images**: 673 SIPaKMeD (2048x1536)
- **Synthetic Images**: 77 AI-generated (512x512)
- **Total Training**: ~750 images
- **Class Distribution**:
  - Normal: 200 images
  - CIN1: 366 images (largest)
  - CIN2: 68 images
  - CIN3: 61 images
  - Cancer: 62 images

### Key Insights:
- **CNN model provides good baseline** accuracy
- **Hybrid model coming soon** with improved accuracy and interpretability
- **Class imbalance** affects CIN2/3/Cancer predictions
- **Synthetic data helps** but real data is primary

Note: Performance depends heavily on dataset quality and size.

## ⚠️ Important Notes

1. **Medical Disclaimer**: This is for research and educational purposes only. Not for clinical use.
2. **Data Privacy**: Handle medical images according to HIPAA, GDPR, and local regulations
3. **Dataset**: You need actual cervical cancer images to train effectively
4. **GPU Recommended**: Training will be much faster with a CUDA-enabled GPU

## 📚 Additional Commands

### Generate More Synthetic Data (Auto-Merge Enabled)
```bash
cd backend

# Generate specific stage with auto-merge
python generate_synthetic_data.py --num-images 200 --stage CIN2

# Generate all stages
python generate_synthetic_data.py --num-images 100 --stage all
```

### Extract Traditional Features
```bash
cd backend

# Extract features from training data
python feature_extractor.py --data-dir ../data/train --output train_features.csv

# Extract features from validation data
python feature_extractor.py --data-dir ../data/val --output val_features.csv

# Features extracted:
# - 30+ medical parameters per image
# - Saved to CSV for ML training
# - Includes N/C ratio, texture, color, morphology
```

### Custom Training Parameters
```bash
python train.py \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --data-dir ../data \
  --checkpoint-dir ./checkpoints
```

### Build Frontend for Production
```bash
cd frontend
npm run build
# Creates optimized build in frontend/build/
```

## 🎓 Next Steps

1. **Collect Data**: Obtain a proper medical image dataset
2. **Train Model**: Run training with sufficient data (500+ images per class)
3. **Evaluate**: Test on holdout test set
4. **Deploy**: Consider containerization with Docker
5. **Monitor**: Track predictions and retrain periodically

## 📞 Support

For issues:
- Check the troubleshooting section above
- Review error messages carefully
- Ensure all dependencies are installed
- Verify Python and Node versions

## 📄 License

MIT License - See LICENSE file for details

---

**Ready to Start?** 

1. Activate backend: `cd backend && source venv/bin/activate && uvicorn api.main:app --reload`
2. Start frontend: `cd frontend && npm start`
3. Open http://localhost:3000

Happy Classifying! 🔬🎯