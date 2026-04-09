from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import load_model, get_class_names
from utils import get_transforms

app = FastAPI(title="Cervical Cancer Classification API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
device = None
transform = None
class_names = None

@app.on_event("startup")
async def load_model_on_startup():
    """Load the model when the API starts"""
    global model, device, transform, class_names
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pth")
    
    if os.path.exists(model_path):
        try:
            model = load_model(model_path, device)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting without pre-trained model. Please train the model first.")
    else:
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
    
    # Load transforms and class names
    transform = get_transforms(augment=False)
    class_names = get_class_names()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Cervical Cancer Classification API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "classes": "/classes"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.get("/classes")
async def get_classes():
    """Get available classification classes"""
    return {
        "classes": class_names,
        "descriptions": {
            "Normal": "Healthy cervical tissue",
            "CIN1": "Cervical Intraepithelial Neoplasia Grade 1 (Low-grade)",
            "CIN2": "Cervical Intraepithelial Neoplasia Grade 2 (High-grade)",
            "CIN3": "Cervical Intraepithelial Neoplasia Grade 3 (High-grade)",
            "Cancer": "Invasive cervical cancer"
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict cervical cancer stage from uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results with class probabilities
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first."
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Transform image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Prepare response
        class_probabilities = {
            class_names[i]: float(probabilities[0][i].item()) 
            for i in range(len(class_names))
        }
        
        return JSONResponse(content={
            "success": True,
            "predicted_class": class_names[predicted_class],
            "confidence": float(confidence),
            "probabilities": class_probabilities
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
