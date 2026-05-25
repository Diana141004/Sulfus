from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="ML Prediction API", description="API pentru inferență model ML - Predicție Payment Delay")

# === PARTEA 1: ÎNCĂRCAREA MODELULUI REAL ===
try:
    model = joblib.load("model_xgb_tuned.pkl")
    print("✅ Modelul a fost încărcat cu succes!")
except Exception as e:
    print(f"⚠️ Eroare critică la încărcarea modelului: {e}")
    model = None

# Coloanele exacte pe care modelul le așteaptă (69 features, one-hot encoded)
MODEL_COLUMNS = [
    'account_length', 'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
    'total_eve_minutes', 'total_eve_calls', 'total_night_minutes', 'total_night_calls',
    'total_intl_minutes', 'total_intl_calls', 'number_customer_service_calls',
    'state_AK', 'state_AL', 'state_AR', 'state_AZ', 'state_CA', 'state_CO',
    'state_CT', 'state_DC', 'state_DE', 'state_FL', 'state_GA', 'state_HI',
    'state_IA', 'state_ID', 'state_IL', 'state_IN', 'state_KS', 'state_KY',
    'state_LA', 'state_MA', 'state_MD', 'state_ME', 'state_MI', 'state_MN',
    'state_MO', 'state_MS', 'state_MT', 'state_NC', 'state_ND', 'state_NE',
    'state_NH', 'state_NJ', 'state_NM', 'state_NV', 'state_NY', 'state_OH',
    'state_OK', 'state_OR', 'state_PA', 'state_RI', 'state_SC', 'state_SD',
    'state_TN', 'state_TX', 'state_UT', 'state_VA', 'state_VT', 'state_WA',
    'state_WI', 'state_WV', 'state_WY', 'area_code_area_code_408',
    'area_code_area_code_415', 'area_code_area_code_510', 'international_plan_no',
    'international_plan_yes', 'voice_mail_plan_no', 'voice_mail_plan_yes'
]


# === PARTEA 2: STRUCTURA DATELOR (Request Body) ===
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


# === PARTEA 3: ENDPOINT-URILE ===
@app.post("/predict")
async def face_predictie(data: InputData):
    """
    Acest endpoint primește datele în format JSON, le convertește în DataFrame,
    aplică one-hot encoding identic cu antrenarea și returnează predicția modelului.
    """
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Modelul ML nu este încărcat pe server. Asigură-te că fișierul 'model_xgb.pkl' se află în același folder."
        )

    try:
        # 1. Transformăm datele validate de Pydantic într-un DataFrame Pandas
        date_dict = data.model_dump()
        df_input = pd.DataFrame([date_dict])

        # 2. Eliminare features redundante (charge = minutes * rate fix)
        cols_to_drop = ['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge']
        df_input = df_input.drop(columns=[c for c in cols_to_drop if c in df_input.columns])

        # 3. Aplicăm One-Hot Encoding pe variabilele categoriale (identic cu antrenarea)
        categorical_features = ['state', 'area_code', 'international_plan', 'voice_mail_plan']
        df_input = pd.get_dummies(df_input, columns=categorical_features)

        # 4. Aliniem coloanele exact cu cele așteptate de model (adăugăm coloane lipsă cu 0)
        df_input = df_input.reindex(columns=MODEL_COLUMNS, fill_value=0)

        # 5. Asigurăm tipurile corecte (bool -> int pentru XGBoost)
        df_input = df_input.astype(float)

        # 6. Trimitem datele către model pentru predicție
        predictie = model.predict(df_input)
        th = 0.7680
        pred = (predictie >= th).astype(int)
        rezultat_final = str(pred[0])

        return {
            "status": "success",
            "prediction": rezultat_final,
            "interpretation": "Clientul RISCĂ întârziere la plată" if rezultat_final == "yes" else "Clientul NU riscă întârziere la plată"
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
        "model_loaded": model is not None
    }
