# 📡 Predicție Payment Delay – Telecomunicații

Proiect de Machine Learning pentru predicția întârzierilor de plată ale clienților unei companii de telecomunicații, dezvoltat ca parte a procesului de selecție internship.

## 🎯 Obiectiv

Clasificare binară: **va întârzia clientul plata?** (`payment_delay`: yes/no)

---

## 📊 Rezultate (Partea 1 – Antrenare)

| Model | AUC | F1 | Recall | Precision | Threshold |
|-------|-----|-----|--------|-----------|-----------|
| Logistic Regression | 81.59% | 46.04% | 73.49% | 33.52% | - |
| SVM (RBF) | 90.54% | 65.96% | 74.70% | 59.05% | - |
| Random Forest | 93.20% | 59.50% | 43.37% | 94.74% | - |
| XGBoost | 92.24% | 83.23% | 80.72% | 85.90% | - |
| **XGBoost Tuned (best F1)** | **92.77%** | **85.90%** | **80.72%** | **91.78%** | **0.768** |
| XGBoost Tuned (high Recall) | 92.77% | 84.02% | 85.54% | 82.56% | 0.6843 |
| CatBoost | 92.68% | 83.13% | 83.13% | 83.13% | - |
| CNN 1D | 75.88% | 11.11% | 6.02% | 71.43% | - |

**🏆 Best model: XGBoost Tuned** – optimizat cu RandomizedSearchCV (80 iterații, 5-fold CV) + threshold tuning pe curba Precision-Recall.

---

## 🚀 Partea 2 – FastAPI (Inferență)

### Rulare server
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Endpoints

| Metoda | Endpoint | Descriere |
|--------|----------|-----------|
| GET | `/` | Health check simplu |
| GET | `/health` | Status server + model loaded |
| POST | `/predict` | **Predicție payment delay** |
| GET | `/docs` | Swagger UI (documentație interactivă) |

### Exemplu Request (POST /predict)
```json
{
    "state": "OH",
    "account_length": 107,
    "area_code": "area_code_415",
    "international_plan": "no",
    "voice_mail_plan": "yes",
    "number_vmail_messages": 26,
    "total_day_minutes": 161.6,
    "total_day_calls": 123,
    "total_day_charge": 27.47,
    "total_eve_minutes": 195.5,
    "total_eve_calls": 103,
    "total_eve_charge": 16.62,
    "total_night_minutes": 254.4,
    "total_night_calls": 103,
    "total_night_charge": 11.45,
    "total_intl_minutes": 13.7,
    "total_intl_calls": 3,
    "total_intl_charge": 3.7,
    "number_customer_service_calls": 1
}
```

### Exemplu Response
```json
{
    "prediction": {
        "label": "payment_delay",
        "value": 0,
        "description": "Clientul NU riscă întârziere la plată"
    },
    "confidence": {
        "probability": 0.1234,
        "threshold_used": 0.768,
        "threshold_mode": "best_f1"
    },
    "model_info": {
        "name": "XGBoost Tuned",
        "version": "1.0",
        "metrics": {"f1": 0.859, "precision": 0.9178, "recall": 0.8072}
    }
}
```

---

## 🤖 Partea 3 – FastMCP (Server MCP)

Modelul ML este expus ca **Tool MCP** apelabil dinamic de un LLM (Claude, GPT, etc.) prin protocolul Model Context Protocol.

### Rulare server MCP
```bash
pip install fastmcp
python mcp_server.py                          # stdio (pentru Claude Desktop)
fastmcp dev inspector mcp_server.py           # inspector web (dezvoltare)
```

### Componente MCP

| Tip | Nume | Descriere |
|-----|------|-----------|
| **Tool** | `predict_payment_delay` | Predicție payment delay cu threshold optimizat |
| Resource | `stats://dataset` | Statistici generale despre dataset |
| Resource | `stats://model-comparison` | Tabel comparare toate modelele |
| Prompt | `interpret_prediction` | Template LLM pentru interpretare rezultat |
| Prompt | `analyze_customer` | Template rapid analiză client |

### Exemplu apel Tool (de către un LLM)
```
"Analizează acest client: stat OH, 161 minute/zi, 1 apel customer service, fără plan internațional. Va întârzia plata?"
```

LLM-ul apelează automat `predict_payment_delay` și primește:
```json
{
    "prediction": 0,
    "probability": 0.7228,
    "threshold_used": 0.768,
    "threshold_mode": "best_f1",
    "description": "Clientul NU riscă întârziere la plată"
}
```

### Verificare funcționare
```bash
fastmcp list mcp_server.py    # listează tool-urile disponibile
```

---

## 🛠️ Tehnologii

- **Python 3.10+**
- FastAPI + Uvicorn
- FastMCP (Model Context Protocol)
- scikit-learn, XGBoost, CatBoost, TensorFlow/Keras
- Pandas, NumPy

## 📁 Structura proiectului

```
├── train.py                    # Partea 1: Pipeline complet de antrenare
├── main.py                     # Partea 2: FastAPI server pentru inferență
├── mcp_server.py               # Partea 3: Server FastMCP (Tool MCP)
├── telecomunicatii.csv         # Dataset (3333 clienți, 20 features)
├── requirements.txt            # Dependințe Python
├── .gitignore
├── models/
│   ├── model_xgb_tuned.joblib  # ⭐ Best model (XGBoost fine-tuned)
│   ├── xgb_thresholds.joblib   # Threshold-uri optime
│   ├── preprocessor.joblib     # ColumnTransformer (scaler + encoder)
│   ├── model_xgb.joblib        # XGBoost default
│   ├── model_lr.joblib         # Logistic Regression
│   ├── model_svm.joblib        # SVM
│   ├── model_rf.joblib         # Random Forest
│   ├── model_catboost.cbm      # CatBoost
│   ├── model_cnn.keras         # CNN 1D
│   └── rezultate_comparare.json
└── README.md
```

---

## 📝 Metodologie

1. **Preprocesare**: StandardScaler (numeric) + OneHotEncoder (categorial)
2. **Feature Engineering**: Eliminare features cu multicolinearitate perfectă (charge = minutes × rate)
3. **Modele testate**: LR, SVM, RF, XGBoost, CatBoost, CNN 1D
4. **Fine-tuning**: RandomizedSearchCV (80 iterații, 5-fold stratified CV, optimizat pe recall)
5. **Threshold tuning**: Analiza curbei Precision-Recall pentru a maximiza F1 sau a asigura Recall ≥ 85%
6. **Evaluare**: Test set 20% stratificat (metrici: AUC, F1, Recall, Precision)

## 🔑 Threshold-uri optime (hardcoded)

```python
BEST_THRESHOLD_F1 = 0.768          # F1=85.9%, Recall=80.72%, Precision=91.78%
BEST_THRESHOLD_HIGH_RECALL = 0.6843  # F1=84.02%, Recall=85.54%, Precision=82.56%
```

---

## 👥 Echipa

Proiect dezvoltat pentru procesul de selecție internship.
