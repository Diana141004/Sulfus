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
| **XGBoost Tuned (best F1)** | **92.77%** | **85.35%** | **80.72%** | **90.54%** | **0.756** |
| XGBoost Tuned (high Recall) | 92.77% | 66.36% | 87.95% | 53.28% | 0.473 |
| CatBoost | 92.68% | 83.13% | 83.13% | 83.13% | - |
| CNN 1D | 84.91% | 42.02% | 30.12% | 69.44% | - |

**🏆 Best model: XGBoost Tuned** – optimizat cu RandomizedSearchCV (80 iterații, 5-fold CV) + threshold tuning pe **Out-of-Fold predictions** (fără data leakage).

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
| GET | `/health` | Status server + model loaded + thresholds |
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
        "threshold_used": 0.756,
        "threshold_mode": "best_f1"
    },
    "model_info": {
        "name": "XGBoost Tuned",
        "version": "2.0",
        "threshold_method": "Out-of-Fold (no data leakage)",
        "metrics_on_test": {"f1": 0.8535, "precision": 0.9054, "recall": 0.8072, "auc": 0.9277}
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
| **Tool** | `predict_payment_delay` | Predicție payment delay cu threshold optimizat (OOF) |
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
    "threshold_used": 0.756,
    "threshold_mode": "best_f1",
    "description": "Clientul NU riscă întârziere la plată",
    "threshold_method": "Out-of-Fold (no data leakage)"
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
├── telecomunicatii.csv         # Dataset (3000 clienți, 20 features)
├── requirements.txt            # Dependințe Python
├── .gitignore
├── models/
│   ├── model_xgb_tuned.joblib  # ⭐ Best model (XGBoost fine-tuned)
│   ├── xgb_thresholds.joblib   # Threshold-uri optime (calculate pe OOF)
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

1. **Preprocesare**: StandardScaler (numeric) + OneHotEncoder (categorial) via `ColumnTransformer`
2. **Feature Engineering**: Eliminare features cu multicolinearitate perfectă (charge = minutes × rate)
3. **Modele testate**: LR, SVM, RF, XGBoost, CatBoost, CNN 1D
4. **Fine-tuning**: RandomizedSearchCV (80 iterații, 5-fold stratified CV, optimizat pe recall)
5. **Threshold tuning**: **Out-of-Fold predictions** pe setul de train (fără data leakage pe test)
6. **Evaluare**: Test set 20% stratificat — metrici nebiased (AUC, F1, Recall, Precision)
7. **Inferență**: Preprocessor identic cu antrenarea (`preprocessor.joblib`), threshold din `xgb_thresholds.joblib`

### ⚠️ Data Leakage Fix (v2.0)

În versiunea anterioară (v1.0), threshold-ul era calculat direct pe test set, ceea ce introducea un **bias de evaluare** — metrici raportate ușor optimiste. 

**Soluția (v2.0):** Threshold-urile sunt acum calculate pe **Out-of-Fold predictions** din cross-validation pe setul de train. Modelul final este antrenat pe tot train-ul, iar evaluarea pe test este complet nebiased.

De asemenea, inferența (FastAPI + MCP) folosește acum `preprocessor.joblib` (StandardScaler + OneHotEncoder identic cu antrenarea), nu `pd.get_dummies()` manual.

## 🔑 Threshold-uri optime (din `models/xgb_thresholds.joblib`)

```python
# Încărcate dinamic la inferență — NU hardcodate
thresholds = joblib.load('models/xgb_thresholds.joblib')
# {'best_f1': 0.756, 'high_recall': 0.473}
```

| Mode | Threshold | F1 | Recall | Precision | Când se folosește |
|------|-----------|-----|--------|-----------|-------------------|
| `best_f1` | 0.756 | 85.35% | 80.72% | 90.54% | Default — echilibru optim |
| `high_recall` | 0.473 | 66.36% | 87.95% | 53.28% | Când vrei să prinzi cât mai mulți clienți cu risc |

---

## 👥 Echipa

Proiect dezvoltat pentru procesul de selecție internship.
