from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="İkinci El Araç Fiyat Tahmin API",
    description="Eğitilmiş Random Forest Modeli ile Araç Fiyat Tahmini Servisi",
    version="1.0"
)

# Model ve Özellik Listesinin Yolları
# Çalışma dizinine göre esnek dosya yolları
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_car_price_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "model_features.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("Model veya özellik listesi dosyası bulunamadı!")

model = joblib.load(MODEL_PATH)
model_features = joblib.load(FEATURES_PATH)

# Kullanıcıdan Alınacak Veri Formatı
class CarInput(BaseModel):
    company: str
    fuel_type: str
    year: int
    kms_driven: float

@app.get("/")
def home():
    return {"status": "online", "message": "Araç Fiyat Tahmin API Servisi Çalışıyor!"}

@app.post("/predict")
def predict_price(car: CarInput):
    try:
        current_year = 2026
        car_age = current_year - car.year
        if car_age < 0:
            car_age = 0
            
        mileage_per_year = car.kms_driven / (car_age + 1)

        input_data = {
            'company': car.company,
            'fuel_type': car.fuel_type,
            'car_age': car_age,
            'kms_driven': car.kms_driven,
            'mileage_per_year': mileage_per_year
        }

        df_input = pd.DataFrame([input_data])
        df_encoded = pd.get_dummies(df_input, columns=['company', 'fuel_type'])

        df_final = df_encoded.reindex(columns=model_features, fill_value=0)

        # Logaritmik tahmini al ve expm1 ile TL'ye çevir
        log_pred = model.predict(df_final)[0]
        actual_pred = float(np.expm1(log_pred))

        return {
            "status": "success",
            "predicted_price_tl": round(actual_pred, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))