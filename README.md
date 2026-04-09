# Cervical Cancer Stage Classification System

An advanced cervical cancer stage classification system using **Deep Learning CNN** for accurate diagnosis, with GenAI for synthetic data augmentation.

## 🎯 Features

### **Three Classification Approaches:**

1. **CNN Deep Learning Model**
   - 6 convolutional layers with batch normalization
   - Learns features automatically from raw pixels
   - 85-88% accuracy on real medical images

2. **Traditional ML with Hand-Crafted Features**
   - Extracts 30+ medical parameters (N/C ratio, cell size, texture, etc.)
   - Clinically validated features used by pathologists
   - Interpretable and explainable predictions
   - 75-80% accuracy

3. **Hybrid Model (CNN + Traditional Features)** ⭐ **COMING SOON**
   - Being redesigned for optimal CPU performance
   - Will combine deep learning with medical expertise
   - Target: 90-95% accuracy with feature importance analysis

### **Additional Features:**
- ✅ **Auto-Merge Synthetic Data**: Generated images automatically added to training
- ✅ **GenAI Data Synthesis**: Stable Diffusion for data augmentation
- ✅ **Multi-stage Classification**: Normal, CIN1, CIN2, CIN3, Cancer
- ✅ **Government Healthcare Portal UI**: Professional frontend interface
- ✅ **REST API**: FastAPI backend for model serving
- ✅ **Feature Importance Analysis**: Know which parameters drive predictions

## 🏗️ System Architecture

```
├── backend/
│   ├── models/
│   │   ├── cnn_model.py           # Pure CNN classifier
│   │   ├── genai_model.py         # Stable Diffusion generator
│   ├── api/                       # FastAPI endpoints
│   ├── utils/                     # Data loaders & transforms
│   ├── train.py                   # Train CNN model
│   ├── feature_extractor.py       # Extract medical features
│   └── generate_synthetic_data.py # Auto-merge synthetic images
├── frontend/
│   └── src/
│       ├── App.js                 # Healthcare portal UI
│       ├── IndiaMap.js            # Cancer statistics visualization
│       └── Chatbot.js             # Healthcare assistant
└── data/
    ├── train/                     # Training images (real + synthetic)
    ├── val/                       # Validation images
    ├── test/                      # Test images
    └── synthetic/                 # Generated images
```

## Installation

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Mac/Linux
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

## 🚀 Usage

### **Option 1: Train CNN Model (Standard Deep Learning)**

```bash
cd backend
python train.py --epochs 50 --batch-size 32 --data-dir ../data
```

### **Option 2: Train Hybrid Model** ⭐ **COMING SOON**

Hybrid model is being redesigned for optimal CPU performance. Will be available soon with improved accuracy and efficiency.

### **Generate Synthetic Data (Auto-Merge Enabled)**

```bash
cd backend

# Generate 50 CIN2 images - automatically merges to training folder
python generate_synthetic_data.py --num-images 50 --stage CIN2

# Generate for all stages
python generate_synthetic_data.py --num-images 100 --stage all
```

**Features:**
- ✅ Skips existing images (generates only new ones)
- ✅ Auto-merges new images to `data/train/` folders
- ✅ Sequential numbering (CIN2_1.png, CIN2_2.png, etc.)

### **Extract Traditional Features Only**

```bash
cd backend
python feature_extractor.py --data-dir ../data/train --output train_features.csv
```

Extracts 30+ parameters:
- Morphological: cell_area, compactness, aspect_ratio
- Nucleus: N/C ratio, nucleus_irregularity
- Texture: GLCM contrast, homogeneity, energy
- Color: RGB/HSV statistics

### Start Backend

```bash
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend

```bash
cd frontend
npm start
```

## 📊 Model Performance Comparison

| Model | Train Acc | Val Acc | Test Acc | Interpretability |
|-------|-----------|---------|----------|------------------|
| Traditional ML (Random Forest) | 80% | 75% | 73% | ⭐⭐⭐⭐⭐ |
| CNN (Deep Learning) | 95% | 88% | 86% | ⭐⭐ |
| **Hybrid (CNN + Features)** | **Coming Soon** | - | - | ⭐⭐⭐⭐ |

### Training Data:
- **Real Images**: 673 SIPaKMeD dataset (2048x1536 resolution)
- **Synthetic Images**: 77 AI-generated (512x512 resolution)
- **Total**: ~750 training images
- **Distribution**: Normal (200), CIN1 (366), CIN2 (68), CIN3 (61), Cancer (62)

### Key Medical Features Used:
- **N/C Ratio**: 0.1-0.3 (Normal) → 0.7-0.9 (Cancer)
- **Cell Compactness**: 0.8-1.0 (Normal) → 0.3-0.5 (Cancer)
- **Nucleus Irregularity**: High score = irregular shape
- **GLCM Texture**: Measures chromatin pattern chaos

## Classification Stages

1. **Normal**: Healthy cervical tissue
2. **CIN1**: Cervical Intraepithelial Neoplasia Grade 1 (Low-grade)
3. **CIN2**: Cervical Intraepithelial Neoplasia Grade 2 (High-grade)
4. **CIN3**: Cervical Intraepithelial Neoplasia Grade 3 (High-grade)
5. **Cancer**: Invasive cervical cancer

## 🛠️ Technologies

### Backend:
- **Deep Learning**: PyTorch, CNN (6 conv layers)
- **Traditional ML**: scikit-image, OpenCV, pandas
- **Feature Extraction**: GLCM, LBP, morphological analysis
- **GenAI**: Stable Diffusion v1.5 (Hugging Face Diffusers)
- **API**: FastAPI, Uvicorn

### Frontend:
- **Framework**: React 18.2.0
- **UI**: Healthcare portal with government styling
- **Visualization**: India cancer statistics map
- **Features**: Image upload, chatbot, scheme information

### Medical Parameters:
- **Morphological**: Cell area, perimeter, compactness, solidity
- **Nucleus**: N/C ratio, irregularity, area
- **Texture**: GLCM (contrast, homogeneity, energy), LBP
- **Color**: RGB/HSV statistics

## License

MIT License
