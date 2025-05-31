"""
API FastAPI que serve dois modelos Keras:
- catdog_model.h5   → gato × cachorro
- orange_model.h5   → laranjas (fresh, rotten, sweet)

Requisitos:
    pip install "fastapi[all]" pillow tensorflow keras==2.15.0  # ou versão compatível do seu .h5
"""

import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import uvicorn

# --- Config --------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent      #  <projeto>/
MODELS_DIR = BASE_DIR / "modelos"

models_cfg = {
    "catdog": {
        "weights": MODELS_DIR / "catdog_model.h5",
        "labels": ["cat", "dog"]
    },
    "orange": {
        "weights": MODELS_DIR / "orange_model.h5",
        "labels": ["fresh_orange", "rotten_orange", "sweet_orange"]
    }
}

# --- Load models once at startup ----------------------------------------
loaded_models = {}
for m_id, cfg in models_cfg.items():
    print(f"🔄  Carregando '{m_id}' de {cfg['weights']}")
    loaded_models[m_id] = tf.keras.models.load_model(cfg["weights"])
    #  👆 se o arquivo não existir ou estiver corrompido, erro aparecerá aqui
print("✅  Modelos carregados.")

# --- FastAPI -------------------------------------------------------------
app = FastAPI(title="Classificador de Imagens")

# Habilite CORS se index.html for servido por outro host/porta
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

def preprocess_pil(img: Image.Image, target_hw: tuple[int, int]) -> np.ndarray:
    """Redimensiona + normaliza [0-1] e devolve shape (1, h, w, 3)."""
    img = img.resize(target_hw).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str = Form(..., description="catdog ou orange")
):
    if model_id not in loaded_models:
        return JSONResponse({"error": f"Modelo '{model_id}' inexistente."}, status_code=400)

    # 1. Lê imagem
    img_bytes = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:
        return JSONResponse({"error": f"Imagem inválida: {exc}"}, status_code=400)

    # 2. Pre-proc
    model = loaded_models[model_id]
    target_h, target_w = model.input_shape[1:3]
    x = preprocess_pil(pil_img, (target_w, target_h))

    # 3. Predição
    preds = model.predict(x, verbose=0)[0]          # 1-D array
    labels = models_cfg[model_id]["labels"]
    idx = int(np.argmax(preds))

    return {
        "label": labels[idx],
        "confidence": float(preds[idx]),
        "probs": {labels[i]: float(preds[i]) for i in range(len(labels))}
    }

# --- CLI -----------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
