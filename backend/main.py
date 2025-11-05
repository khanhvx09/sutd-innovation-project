from io import BytesIO
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from pydantic import BaseModel


tags_metadata = [
    {"name": "system", "description": "System and health endpoints."},
    {"name": "inference", "description": "Image-based emotion detection endpoints."},
]

app = FastAPI(
    title="Emotion Detection API",
    version="1.0.0",
    description="Upload a face image to get an emotion prediction using a ViT model.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=tags_metadata,
)

# Enable CORS (relaxed for development; tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Globals initialized at startup
processor: ViTImageProcessor | None = None
model: ViTForImageClassification | None = None
device: torch.device | None = None


@app.on_event("startup")
def load_model():
    """Load the ViT model and image processor once at startup."""
    global processor, model, device

    # Select device: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load pretrained artifacts
    model_name = "abhilash88/face-emotion-detection"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()


@app.get("/", tags=["system"], summary="Service banner")
def read_root():
    return {"message": "Emotion Detection API is running"}


@app.get("/healthz", tags=["system"], summary="Health check")
def healthz():
    return {"status": "ok"}


class ProbabilityItem(BaseModel):
    label: str
    probability: float


class EmotionPredictionResponse(BaseModel):
    emotion: str
    confidence: float



def _predict_emotion(pil_image: Image.Image):
    """Internal helper to run model inference on a PIL image."""
    assert processor is not None and model is not None and device is not None, "Model not loaded"

    # Preprocess and move tensors to device
    inputs = processor(pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}


    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()

    # Emotion classes
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_emotion = emotions[predicted_class]
    confidence = predictions[0][predicted_class].item()


    return {
        "emotion": predicted_emotion,
        "confidence": confidence
    }


@app.post(
    "/predict",
    tags=["inference"],
    summary="Predict emotion from an uploaded image",
    response_model=EmotionPredictionResponse,
)
async def predict(file: UploadFile = File(..., description="Image file (jpeg/png) to analyze")):
    """Accept an image file and return the predicted emotion and confidence."""
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        content = await file.read()
        pil_image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Unsupported or corrupted image file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read image: {e}")

    try:
        result = _predict_emotion(pil_image)
    except AssertionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return result