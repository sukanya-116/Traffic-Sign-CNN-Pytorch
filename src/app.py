import os
import io
import numpy as np
import onnxruntime as ort
from keras_image_helper import create_preprocessor
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
from PIL import Image
import io

# -----------------------------
# Paths
# -----------------------------
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
onnx_model = os.path.join(
    repo_path,
    "model",
    "traffic_sign_efficientnet_b0.onnx"
)

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="traffic-sign-classifier")

# -----------------------------
# Preprocessing 
# -----------------------------

def preprocess_pytorch_style(img):
    img = img.resize((224, 224))
    X = np.array(img, dtype=np.float32)  # (H, W, C)

    X = X / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    X = (X - mean) / std
    X = np.transpose(X, (2, 0, 1))  # HWC → CHW
    X = np.expand_dims(X, axis=0)   # → NCHW

    return X

preprocessor = create_preprocessor(
    preprocess_pytorch_style,
    target_size=(224, 224)
)

# -----------------------------
# ONNX Runtime
# -----------------------------
session = ort.InferenceSession(
    onnx_model,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# -----------------------------
# Classes
# -----------------------------
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

NUM_CLASSES = 43
TOP_K = 5

# -----------------------------
# Softmax
# -----------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# -----------------------------
# Response Model
# -----------------------------
class PredictResponse(BaseModel):
    predictions: dict[str, float]
    top_class: str
    top_probability: float

# -----------------------------
# Prediction logic
# -----------------------------
def predict(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    X = preprocessor.preprocess(image)

    logits = session.run([output_name], {input_name: X})[0][0]
    probs = softmax(logits)

    top_indices = np.argsort(probs)[-TOP_K:][::-1]

    predictions = {
        classes[i]: float(probs[i]) for i in top_indices
    }

    top_idx = int(top_indices[0])

    return predictions, classes[top_idx], float(probs[top_idx])

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "Traffic Sign Classification Service"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        predictions, top_class, top_prob = predict(image_bytes)

        return PredictResponse(
            predictions=predictions,
            top_class=top_class,
            top_probability=top_prob
        )

    except Exception as e:
        return {
            "error": str(e)
        }

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
