# import joblib
# import pandas as pd
# from fastapi import FastAPI
# from pydantic import BaseModel
# from pathlib import Path

# MODEL_PATH = Path("pipeline/full_pipeline.joblib")

# app = FastAPI(
#     title="Predict your fat percentage",
#     version="1.0"
# )

# pipeline = None

# # -------------------------
# # Load model on startup
# # -------------------------
# @app.on_event("startup")
# def load_model():
#     global pipeline
#     try:
#         pipeline = joblib.load(MODEL_PATH)
#         print("✅ Model loaded successfully")
#     except Exception as e:
#         raise RuntimeError(f"❌ Failed to load model: {e}")

# # -------------------------
# # Schemas
# # -------------------------
# class UserInput(BaseModel):
#     age: int
#     gender: str
#     weight: float
#     height: float
#     max_bpm: int
#     avg_bpm: int
#     resting_bpm: int
#     session_duration: float
#     calories_burned: float
#     workout_type: str
#     water_intake: float
#     workout_frequency: int
#     experience_level: int
#     bmi: float
#     age_category: str
#     weight_category: str
#     bmi_category: str

# class PredictionOutput(BaseModel):
#     fat_percentage: float
#     category: str
#     meaning: str
#     recommendation: str


# # -------------------------
# # Health endpoints
# # -------------------------
# @app.get("/")
# def root():
#     return {"status": "running"}

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# # -------------------------
# # Human readable time
# # -------------------------
# def fat_analysis(pred: float, gender: str):
#     gender = gender.lower()

#     if gender == "male":
#         if pred < 6:
#             category = "🩻 Essential Fat"
#             meaning = "Minimal fat necessary for organs"
#         elif pred < 14:
#             category = "🏆 Athlete Level"
#             meaning = "Athlete-level body fat"
#         elif pred < 18:
#             category = "💪 Fitness Level"
#             meaning = "Good physical fitness"
#         elif pred < 25:
#             category = "✅ Healthy Normal"
#             meaning = "Healthy and balanced"
#         elif pred < 30:
#             category = "⚠️ Overfat"
#             meaning = "Above normal fat level"
#         else:
#             category = "🚨 Obese Risk"
#             meaning = "Risk zone for health"
#     else:  # female
#         if pred < 14:
#             category = "🩻 Essential Fat"
#             meaning = "Minimal fat necessary for organs"
#         elif pred < 21:
#             category = "🏆 Athlete Level"
#             meaning = "Athlete-level body fat"
#         elif pred < 25:
#             category = "💪 Fitness Level"
#             meaning = "Good physical fitness"
#         elif pred < 32:
#             category = "✅ Healthy Normal"
#             meaning = "Healthy and balanced"
#         elif pred < 36:
#             category = "⚠️ Overfat"
#             meaning = "Above normal fat level"
#         else:
#             category = "🚨 Obese Risk"
#             meaning = "Risk zone for health"

#     # Advice
#     if pred < 18:
#         advice = "🥗 Improve nutrition balance + increase muscle mass"
#     elif pred < 25:
#         advice = "✅ Maintain current regime + consistent workouts"
#     elif pred < 30:
#         advice = "🔥 Increase cardio and HIIT + drink more water"
#     else:
#         advice = "🚨 Nutrition plan + strict training regimen + monitoring"

#     return category, meaning, advice

# # -------------------------
# # Predict endpoint
# # -------------------------
# @app.post("/predict", response_model=PredictionOutput)
# def predict(data: UserInput):
#     if pipeline is None:
#         raise RuntimeError("Model not loaded")

#     df = pd.DataFrame([data.dict()])

#     pred = float(pipeline.predict(df)[0])

#     category, meaning, advice = fat_analysis(pred, data.gender)

#     return PredictionOutput(
#         fat_percentage=round(pred, 1),
#         category=category,
#         meaning=meaning,
#         recommendation=advice
#     )




import pandas as pd
import torch
import torch.nn as nn
import joblib

from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

DL_MODEL_PATH = Path("model/structured_nn.pth")
ENCODER_PATH = Path("model/encoder.joblib")
SCALER_PATH = Path("model/scaler.joblib")

app = FastAPI(
    title="Predict your fat percentage",
    version="2.0"
)

dl_model = None
encoder = None
scaler = None

# -------------------------
# Neural Network Architecture
# -------------------------
class StructuredNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Load model on startup
# -------------------------
@app.on_event("startup")
def load_model():
    global dl_model, encoder, scaler

    try:
        encoder = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)

        dl_model = StructuredNN(input_dim=17)
        dl_model.load_state_dict(torch.load(DL_MODEL_PATH, map_location="cpu"))
        dl_model.eval()

        print("✅ DL Model loaded successfully")

    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {e}")

# -------------------------
# Schemas
# -------------------------
class UserInput(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    max_bpm: int
    avg_bpm: int
    resting_bpm: int
    session_duration: float
    calories_burned: float
    workout_type: str
    water_intake: float
    workout_frequency: int
    experience_level: int
    bmi: float
    age_category: str
    weight_category: str
    bmi_category: str


class PredictionOutput(BaseModel):
    fat_percentage: float
    category: str
    meaning: str
    recommendation: str


# -------------------------
# Health endpoints
# -------------------------
@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# Fat analysis
# -------------------------
def fat_analysis(pred: float, gender: str):
    gender = gender.lower()

    if gender == "male":
        if pred < 6:
            category = "🩻 Essential Fat"
            meaning = "Minimal fat necessary for organs"
        elif pred < 14:
            category = "🏆 Athlete Level"
            meaning = "Athlete-level body fat"
        elif pred < 18:
            category = "💪 Fitness Level"
            meaning = "Good physical fitness"
        elif pred < 25:
            category = "✅ Healthy Normal"
            meaning = "Healthy and balanced"
        elif pred < 30:
            category = "⚠️ Overfat"
            meaning = "Above normal fat level"
        else:
            category = "🚨 Obese Risk"
            meaning = "Risk zone for health"
    else:
        if pred < 14:
            category = "🩻 Essential Fat"
            meaning = "Minimal fat necessary for organs"
        elif pred < 21:
            category = "🏆 Athlete Level"
            meaning = "Athlete-level body fat"
        elif pred < 25:
            category = "💪 Fitness Level"
            meaning = "Good physical fitness"
        elif pred < 32:
            category = "✅ Healthy Normal"
            meaning = "Healthy and balanced"
        elif pred < 36:
            category = "⚠️ Overfat"
            meaning = "Above normal fat level"
        else:
            category = "🚨 Obese Risk"
            meaning = "Risk zone for health"

    if pred < 18:
        advice = "🥗 Improve nutrition balance + increase muscle mass"
    elif pred < 25:
        advice = "✅ Maintain current regime + consistent workouts"
    elif pred < 30:
        advice = "🔥 Increase cardio and HIIT + drink more water"
    else:
        advice = "🚨 Nutrition plan + strict training regimen + monitoring"

    return category, meaning, advice


# -------------------------
# Predict endpoint
# -------------------------
@app.post("/predict", response_model=PredictionOutput)
def predict(data: UserInput):

    if dl_model is None:
        raise RuntimeError("Model not loaded")

    df = pd.DataFrame([data.dict()])

    # -------- ENCODING --------
    cat_cols = df.select_dtypes(include=['object']).columns

    if len(cat_cols) > 0:
        df[cat_cols] = encoder.transform(df[cat_cols])

    # -------- SCALING --------
    df = scaler.transform(df)

    # -------- TORCH --------
    x = torch.tensor(df, dtype=torch.float32)

    with torch.no_grad():
        pred = float(dl_model(x).item())

    category, meaning, advice = fat_analysis(pred, data.gender)

    return PredictionOutput(
        fat_percentage=round(pred, 1),
        category=category,
        meaning=meaning,
        recommendation=advice
    )