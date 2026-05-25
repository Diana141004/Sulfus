"""
Partea 3: Server FastMCP – Expune modelul ML ca Tool MCP.

Permite unui LLM (Claude, GPT, etc.) să apeleze dinamic predicția payment_delay
prin protocolul MCP (Model Context Protocol).

Rulare:
    python mcp_server.py                    # stdio (pentru Claude Desktop)
    fastmcp dev mcp_server.py               # inspector web (dezvoltare)
    fastmcp run mcp_server.py --transport sse --port 8001  # SSE (pentru OpenWebUI)
"""

from fastmcp import FastMCP
import joblib
import pandas as pd
import numpy as np
import os

# ===========================================================================
# INITIALIZARE SERVER MCP
# ===========================================================================
mcp = FastMCP(
    "Telecom Payment Delay Predictor",
    instructions="Server MCP pentru predicția întârzierilor de plată în telecomunicații. "
                 "Folosește un model XGBoost fine-tuned cu threshold optimizat pe OOF (fără data leakage). "
                 "Apelează tool-ul predict_payment_delay cu datele unui client pentru a obține predicția."
)

# Încărcare model + preprocessor + threshold-uri la startup
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
model = joblib.load(os.path.join(MODEL_DIR, "model_xgb_tuned.joblib"))
preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
thresholds = joblib.load(os.path.join(MODEL_DIR, "xgb_thresholds.joblib"))


# ===========================================================================
# TOOL: predict_payment_delay (OBLIGATORIU)
# ===========================================================================
@mcp.tool()
def predict_payment_delay(
    state: str,
    account_length: int,
    area_code: str,
    international_plan: str,
    voice_mail_plan: str,
    number_vmail_messages: int,
    total_day_minutes: float,
    total_day_calls: int,
    total_eve_minutes: float,
    total_eve_calls: int,
    total_night_minutes: float,
    total_night_calls: int,
    total_intl_minutes: float,
    total_intl_calls: int,
    number_customer_service_calls: int,
    threshold_mode: str = "best_f1"
) -> dict:
    """
    Predicție dacă un client telecom va întârzia plata.

    Modelul folosit: XGBoost Tuned (F1=85.35%, AUC=92.77%).
    Threshold calculat pe Out-of-Fold predictions (fără data leakage).
    threshold_mode='best_f1' maximizează F1-score, 'high_recall' asigură Recall >= 85%.

    Args:
        state: Statul clientului (ex: 'OH', 'CA', 'NY')
        account_length: Numărul de zile de când e client
        area_code: Codul zonei (ex: 'area_code_415', 'area_code_408', 'area_code_510')
        international_plan: 'yes' sau 'no'
        voice_mail_plan: 'yes' sau 'no'
        number_vmail_messages: Număr mesaje voicemail
        total_day_minutes: Total minute ziua
        total_day_calls: Total apeluri ziua
        total_eve_minutes: Total minute seara
        total_eve_calls: Total apeluri seara
        total_night_minutes: Total minute noaptea
        total_night_calls: Total apeluri noaptea
        total_intl_minutes: Total minute internaționale
        total_intl_calls: Total apeluri internaționale
        number_customer_service_calls: Apeluri la customer service
        threshold_mode: 'best_f1' (default) sau 'high_recall'

    Returns:
        Dict cu predicția, probabilitatea și interpretarea.
    """
    # Selectare threshold din fișierul salvat
    th = thresholds.get(threshold_mode, thresholds['best_f1'])

    # Construire DataFrame cu datele clientului (identic cu train.py)
    data = {
        'account_length': [account_length],
        'number_vmail_messages': [number_vmail_messages],
        'total_day_minutes': [total_day_minutes],
        'total_day_calls': [total_day_calls],
        'total_eve_minutes': [total_eve_minutes],
        'total_eve_calls': [total_eve_calls],
        'total_night_minutes': [total_night_minutes],
        'total_night_calls': [total_night_calls],
        'total_intl_minutes': [total_intl_minutes],
        'total_intl_calls': [total_intl_calls],
        'number_customer_service_calls': [number_customer_service_calls],
        'state': [state],
        'area_code': [area_code],
        'international_plan': [international_plan],
        'voice_mail_plan': [voice_mail_plan],
    }

    df_input = pd.DataFrame(data)

    # Asigurăm tipurile categoriale (identic cu train.py)
    categorical_features = ['state', 'area_code', 'international_plan', 'voice_mail_plan']
    for c in categorical_features:
        if c in df_input.columns:
            df_input[c] = df_input[c].astype(str)

    # Aplicăm preprocessorul salvat (StandardScaler + OneHotEncoder)
    # Același ColumnTransformer folosit la antrenare — fără discrepanțe
    X_processed = preprocessor.transform(df_input)

    # Predicție cu threshold optimizat (calculat pe OOF)
    probability = float(model.predict_proba(X_processed)[:, 1][0])
    prediction = int(probability >= th)

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "threshold_used": round(th, 4),
        "threshold_mode": threshold_mode,
        "description": "Clientul RISCĂ întârziere la plată" if prediction == 1
                       else "Clientul NU riscă întârziere la plată",
        "model": "XGBoost Tuned (F1=85.35%, Precision=90.54%, Recall=80.72%)",
        "threshold_method": "Out-of-Fold (no data leakage)"
    }


# ===========================================================================
# RESOURCE: Statistici Dataset (RECOMANDAT)
# ===========================================================================
@mcp.resource("stats://dataset")
def get_dataset_stats() -> str:
    """
    Statistici generale despre datasetul de telecomunicații.
    Un LLM poate consulta aceste date pentru context înainte de a interpreta predicții.
    """
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telecomunicatii.csv")
    df = pd.read_csv(csv_path)

    total = len(df)
    delay_yes = df['payment_delay'].value_counts().get('yes', 0)
    delay_no = df['payment_delay'].value_counts().get('no', 0)
    rate = delay_yes / total * 100

    stats = f"""📊 STATISTICI DATASET TELECOMUNICAȚII
{'='*50}
Total clienți: {total}
Payment delay = YES: {delay_yes} ({rate:.1f}%)
Payment delay = NO: {delay_no} ({100-rate:.1f}%)

📈 Features numerice (medii):
- Account length: {df['account_length'].mean():.0f} zile
- Total day minutes: {df['total_day_minutes'].mean():.1f} min
- Total eve minutes: {df['total_eve_minutes'].mean():.1f} min
- Total night minutes: {df['total_night_minutes'].mean():.1f} min
- Total intl minutes: {df['total_intl_minutes'].mean():.1f} min
- Customer service calls: {df['number_customer_service_calls'].mean():.1f}

📋 Features categoriale:
- International plan YES: {(df['international_plan']=='yes').sum()} ({(df['international_plan']=='yes').mean()*100:.1f}%)
- Voice mail plan YES: {(df['voice_mail_plan']=='yes').sum()} ({(df['voice_mail_plan']=='yes').mean()*100:.1f}%)
- State-uri unice: {df['state'].nunique()}

🏆 Model utilizat: XGBoost Tuned
- AUC: 92.77%
- F1: 85.35% (threshold OOF = {thresholds['best_f1']:.4f})
- Precision: 90.54%
- Recall: 80.72%
- Threshold method: Out-of-Fold (no data leakage)
"""
    return stats


# ===========================================================================
# RESOURCE: Rezultate comparare modele
# ===========================================================================
@mcp.resource("stats://model-comparison")
def get_model_comparison() -> str:
    """Rezultatele comparării tuturor modelelor antrenate."""
    import json
    json_path = os.path.join(MODEL_DIR, "rezultate_comparare.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)

        output = "📊 COMPARARE MODELE ML\n" + "=" * 50 + "\n"
        output += f"{'Model':<30} {'AUC':>7} {'F1':>7} {'Recall':>7} {'Precision':>9}\n"
        output += "-" * 65 + "\n"

        for name, metrics in results.items():
            output += f"{name:<30} {metrics['AUC']*100:>6.2f}% {metrics['F1']*100:>6.2f}% "
            output += f"{metrics['Recall']*100:>6.2f}% {metrics['Precision']*100:>8.2f}%\n"

        return output
    return "Fișierul rezultate_comparare.json nu a fost găsit."


# ===========================================================================
# PROMPT: Interpretare predicție (RECOMANDAT)
# ===========================================================================
@mcp.prompt()
def interpret_prediction(probability: str, prediction: str, customer_info: str) -> str:
    """
    Template de prompt pentru interpretarea unei predicții de payment delay.
    LLM-ul folosește acest template pentru a genera o explicație non-tehnică.

    Args:
        probability: Probabilitatea returnată de model (ex: '0.82')
        prediction: Decizia modelului ('0' = nu riscă, '1' = riscă)
        customer_info: Descriere text a clientului analizat
    """
    pred_label = "VA ÎNTÂRZIA plata" if prediction == "1" else "NU VA ÎNTÂRZIA plata"
    prob_pct = float(probability) * 100
    th = thresholds['best_f1']

    return f"""Ești un analist de date specializat în telecomunicații. Analizează următorul rezultat:

## Rezultat Predicție ML
- **Decizie model**: {pred_label}
- **Probabilitate**: {prob_pct:.1f}%
- **Threshold folosit**: {th*100:.1f}% (optimizat pe Out-of-Fold, fără data leakage)
- **Model**: XGBoost Tuned (F1=85.35%, Precision=90.54%)

## Date Client
{customer_info}

## Sarcina ta:
1. Explică în limbaj simplu ce înseamnă acest rezultat pentru echipa de customer service
2. Identifică factorii de risc principali din datele clientului
3. Sugerează 2-3 acțiuni concrete pe care echipa le poate lua
4. Menționează nivelul de încredere al predicției (probabilitatea vs threshold)

Răspunde concis și orientat pe acțiune."""


# ===========================================================================
# PROMPT: Analiză client
# ===========================================================================
@mcp.prompt()
def analyze_customer(state: str, minutes_day: str, service_calls: str, intl_plan: str) -> str:
    """
    Template rapid pentru analiza unui client.
    Folosește-l înainte de a apela tool-ul de predicție.
    """
    return f"""Analizează profilul unui client telecom:
- Stat: {state}
- Minute/zi: {minutes_day}
- Apeluri customer service: {service_calls}
- Plan internațional: {intl_plan}

Pe baza acestor date:
1. Folosește tool-ul predict_payment_delay pentru a obține predicția
2. Consultă resursa stats://dataset pentru context
3. Interpretează rezultatul pentru echipa de retenție clienți"""


# ===========================================================================
# PORNIRE SERVER
# ===========================================================================
if __name__ == "__main__":
    mcp.run()
