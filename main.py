import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="WoofWoof ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
model = None
preprocessor = None
label_encoder = None

@app.on_event("startup")
def load_models():
    global model, preprocessor, label_encoder
    model_dir = Path(os.environ.get("MODEL_DIR", "trained_model"))
    model        = joblib.load(model_dir / "voting_clf_model.pkl")
    preprocessor = joblib.load(model_dir / "preprocessor.pkl")
    label_encoder= joblib.load(model_dir / "label_encoder.pkl")
    print("Models loaded successfully!")

ALL_SYMPTOMS = [
    'Fever','Lethargy','Appetite Loss','Vomiting','Diarrhea','Coughing',
    'Labored Breathing','Lameness','Skin Lesions','Nasal Discharge',
    'Eye Discharge','Excessive Thirst','Excessive Urination','Weight Loss',
    'Weight Gain','Swelling','Lumps','Pale Gums','Behavior Change',
    'Muscle Wasting','Head Pressing','Pot-belly','Excessive Scratching',
    'Itching','Hair Loss','Seizures','Foaming of the mouth',
    'Sensitivity to light or sound','Unable to eat / drink'
]

class PredictRequest(BaseModel):
    breed: str
    age: float
    weight: float
    diet: str
    environment: str
    weather: str
    kapon: str
    vaccination: str
    symptoms: List[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    features = {
        'Age': req.age,
        'Weight': req.weight,
        'Breed': req.breed,
        'Diet': req.diet,
        'Environment': req.environment,
        'Weather': req.weather,
        'Kapon': req.kapon,
        'Vaccination_Status': req.vaccination,
    }
    symptoms_set = set(req.symptoms)
    for s in ALL_SYMPTOMS:
        features[s] = 'Yes' if s in symptoms_set else 'No'

    input_df = pd.DataFrame([features])
    
    for col in preprocessor.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 'No'
    input_df = input_df[preprocessor.feature_names_in_]

    processed = preprocessor.transform(input_df)
    if hasattr(processed, "toarray"):
        processed = processed.toarray()

    prediction   = model.predict(processed)
    proba        = model.predict_proba(processed)
    confidence   = float(np.max(proba))
    disease_name = label_encoder.inverse_transform([prediction[0]])[0]

    all_idx = np.argsort(proba[0])[::-1]
    all_diseases = label_encoder.inverse_transform(all_idx)
    all_confs    = proba[0][all_idx]

    filtered = [
        {"disease": d, "confidence": round(min(float(c)*100, 99.9), 2)}
        for d, c in zip(all_diseases, all_confs)
        if c > 0.0001
    ]

    return {
        "primary_prediction": disease_name,
        "confidence": round(min(confidence*100, 99.9), 2),
        "all_predictions": filtered
    }