"""
main.py — FastAPI inference server for EfficientNetHybrid cervical cancer classifier.

Key fix vs previous version:
  EfficientNetHybrid.forward() requires TWO inputs: (image_tensor, feature_tensor).
  The old server only passed the image, causing a TypeError at inference time.
  This version extracts medical features from the uploaded image and passes both.
"""

import io
import os
import sys
from pathlib import Path

# ── Path setup (must happen before ANY local imports) ─────────────────────────
# Use Path(__file__).resolve() so this works regardless of cwd, which changes
# in uvicorn --reload spawned worker processes.
_HERE    = Path(__file__).resolve().parent          # …/backend/api
_BACKEND = _HERE.parent                             # …/backend
_SCRIPTS = _BACKEND / "scripts"                     # …/backend/scripts (feature_extractor.py lives here)
_PROJECT = _BACKEND.parent                          # …/cervical-cancer-classifier-local

for p in (_HERE, _BACKEND, _SCRIPTS, _PROJECT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from models import load_model, get_class_names
from utils  import get_transforms

# Try every plausible import path for feature_extractor
try:
    from feature_extractor import extract_medical_features
except ImportError:
    try:
        from backend.feature_extractor import extract_medical_features
    except ImportError as exc:
        raise ImportError(
            "Cannot find feature_extractor.py. "
            f"Searched sys.path: {sys.path}"
        ) from exc

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Cervical Cancer Classification API — EfficientNetHybrid")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model            = None
device           = None
transform        = None
class_names      = None
model_load_error = None


@app.on_event("startup")
async def load_model_on_startup():
    global model, device, transform, class_names, model_load_error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[startup] device: {device}")

    model_path = str(_BACKEND / "checkpoints" / "best_hybrid_model.pth")
    print(f"[startup] looking for checkpoint: {model_path}")

    if not os.path.exists(model_path):
        model_load_error = (
            f"Checkpoint not found: {model_path}. "
            "Run train_hybrid.py first to generate it."
        )
        print(f"[startup] ERROR: {model_load_error}")
    else:
        try:
            model = load_model(model_path, device)
            model_load_error = None
            print("[startup] model loaded OK")
        except Exception as exc:
            model_load_error = f"Model file found but failed to load: {exc}"
            print(f"[startup] ERROR: {model_load_error}")

    try:
        transform   = get_transforms(augment=False)
        class_names = get_class_names()
        print(f"[startup] classes: {class_names}")
    except Exception as exc:
        raise RuntimeError(f"Could not load transforms/class names: {exc}") from exc


@app.get("/")
async def root():
    return {
        "message": "Cervical Cancer Classification API",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "endpoints": {"/predict": "POST", "/health": "GET", "/classes": "GET"},
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_error": model_load_error,
        "device": str(device),
    }


@app.get("/classes")
async def get_classes_endpoint():
    return {
        "classes": class_names,
        "descriptions": {
            "Normal": "Healthy cervical tissue",
            "CIN1":   "Cervical Intraepithelial Neoplasia Grade 1 (Low-grade)",
            "CIN2":   "Cervical Intraepithelial Neoplasia Grade 2 (High-grade)",
            "CIN3":   "Cervical Intraepithelial Neoplasia Grade 3 (High-grade)",
            "Cancer": "Invasive cervical cancer",
        },
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict cervical cancer stage from an uploaded image.
    Passes both the image tensor AND extracted medical features to the model.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error or "Model not loaded. Check /health for details.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got: {file.content_type}",
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Extract the same 30-dim medical features used during training
        raw_features   = extract_medical_features(pil_image)
        feature_tensor = torch.tensor(raw_features, dtype=torch.float32).unsqueeze(0).to(device)

        # Image tensor
        image_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Inference — model requires BOTH tensors
        with torch.no_grad():
            logits, attn  = model(image_tensor, feature_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence    = float(probabilities[0][predicted_idx].item())

        class_probabilities = {
            class_names[i]: round(float(probabilities[0][i].item()), 6)
            for i in range(len(class_names))
        }

        attn_vals = attn[0].detach().cpu().numpy().tolist()

        return JSONResponse(content={
            "success":         True,
            "predicted_class": class_names[predicted_idx],
            "confidence":      round(confidence, 6),
            "probabilities":   class_probabilities,
            "attention_weights": {
                "cnn":     round(attn_vals[0], 4),
                "medical": round(attn_vals[1], 4),
            },
        })

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing image: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)