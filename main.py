from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="ML Prediction API", description="API pentru inferență model ML - Predicție Payment Delay")

# === ÎNCĂRCAREA MODELULUI + PREPROCESSOR + THRESHOLD-URI ===
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

try:
    model = joblib.load(os.path.join(MODEL_DIR, "model_xgb_tuned.joblib"))
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
    thresholds = joblib.load(os.path.join(MODEL_DIR, "xgb_thresholds.joblib"))
    print("✅ Model, preprocessor și threshold-uri încărcate cu succes!")
    print(f"   Threshold best_f1: {thresholds['best_f1']:.4f}")
    print(f"   Threshold high_recall: {thresholds['high_recall']:.4f}")
except Exception as e:
    print(f"⚠️ Eroare critică la încărcarea modelului: {e}")
    model = None
    preprocessor = None
    thresholds = {'best_f1': 0.5, 'high_recall': 0.5}


# === STRUCTURA DATELOR (Request Body) ===
class InputData(BaseModel):
    state: str
    account_length: int
    area_code: str
    international_plan: str
    voice_mail_plan: str
    number_vmail_messages: int
    total_day_minutes: float
    total_day_calls: int
    total_day_charge: float
    total_eve_minutes: float
    total_eve_calls: int
    total_eve_charge: float
    total_night_minutes: float
    total_night_calls: int
    total_night_charge: float
    total_intl_minutes: float
    total_intl_calls: int
    total_intl_charge: float
    number_customer_service_calls: int


# === ENDPOINT-URILE ===
@app.post("/predict")
async def face_predictie(data: InputData, threshold_mode: str = "best_f1"):
    """
    Primește datele unui client în format JSON, aplică preprocessorul identic cu
    antrenarea (StandardScaler + OneHotEncoder) și returnează predicția modelului.

    threshold_mode: 'best_f1' (default) sau 'high_recall'
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=500,
            detail="Modelul ML sau preprocessorul nu sunt încărcate. Verifică folderul 'models/'."
        )

    try:
        # 1. Transformăm datele validate de Pydantic într-un DataFrame Pandas
        date_dict = data.model_dump()
        df_input = pd.DataFrame([date_dict])

        # 2. Eliminare features redundante (charge = minutes * rate fix)
        # Identic cu train.py
        cols_to_drop = ['total_day_charge', 'total_eve_charge',
                        'total_night_charge', 'total_intl_charge']
        df_input = df_input.drop(columns=[c for c in cols_to_drop if c in df_input.columns])

        # 3. Asigurăm tipurile categoriale (identic cu train.py)
        categorical_features = ['state', 'area_code', 'international_plan', 'voice_mail_plan']
        for c in categorical_features:
            if c in df_input.columns:
                df_input[c] = df_input[c].astype(str)

        # 4. Aplicăm preprocessorul salvat (StandardScaler + OneHotEncoder)
        # Același ColumnTransformer folosit la antrenare — fără data leakage
        X_processed = preprocessor.transform(df_input)

        # 5. Selectare threshold din fișierul salvat
        th = thresholds.get(threshold_mode, thresholds['best_f1'])

        # 6. Predicție cu threshold optimizat (calculat pe OOF, nu pe test)
        prob = float(model.predict_proba(X_processed)[:, 1][0])
        prediction = int(prob >= th)

        return {
            "prediction": {
                "label": "payment_delay",
                "value": prediction,
                "description": "Clientul RISCĂ întârziere la plată" if prediction == 1
                               else "Clientul NU riscă întârziere la plată"
            },
            "confidence": {
                "probability": round(prob, 4),
                "threshold_used": round(th, 4),
                "threshold_mode": threshold_mode
            },
            "model_info": {
                "name": "XGBoost Tuned",
                "version": "2.0",
                "threshold_method": "Out-of-Fold (no data leakage)",
                "metrics_on_test": {
                    "f1": 0.8535,
                    "precision": 0.9054,
                    "recall": 0.8072,
                    "auc": 0.9277
                }
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Eroare la procesarea predicției: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Serverul funcționează. Folosiți POST /predict pentru predicții."}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "thresholds": thresholds
    }
