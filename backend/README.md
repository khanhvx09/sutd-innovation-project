# Emotion Detection API (FastAPI)

A simple FastAPI service that accepts an uploaded image and returns the predicted facial emotion using a ViT model from Hugging Face (`abhilash88/face-emotion-detection`).

## Endpoints

- `GET /` — Service banner
- `GET /healthz` — Health check
- `POST /predict` — Multipart form upload with field name `file` containing an image

### Response (example)

```json
{
  "emotion": "Happy",
  "confidence": 0.97,
  "probabilities": [
    { "label": "Happy", "probability": 0.97 },
    { "label": "Neutral", "probability": 0.02 },
    { "label": "Surprise", "probability": 0.01 }
  ]
}
```

## Setup

Create a virtual environment and install dependencies.

```bash
# From the backend/ directory
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note: Installing PyTorch can vary by platform. If the generic install fails or you want GPU/MPS acceleration, follow the official selector: https://pytorch.org/get-started/locally/

## Run

```bash
# From the backend/ directory with venv activated
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs to try the API via Swagger UI.

## CORS

CORS is permissive in development (allowing all origins). For production, restrict `allow_origins` in `main.py` to your frontend domain(s).
