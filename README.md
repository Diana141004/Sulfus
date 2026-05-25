# 📡 Predicție Payment Delay – Telecomunicații

Proiect de Machine Learning pentru predicția întârzierilor de plată ale clienților unei companii de telecomunicații, dezvoltat ca parte a procesului de selecție internship.

## 🎯 Obiectiv

Clasificare binară: **va întârzia clientul plata?** (`payment_delay`: yes/no)

## 📊 Rezultate

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

## 🛠️ Tehnologii

- **Python 3.10+**
- scikit-learn, XGBoost, CatBoost, TensorFlow/Keras
- Pandas, NumPy

## 📁 Structura proiectului

```
├── train.py                    # Pipeline complet de antrenare
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

## 🚀 Utilizare

### Instalare dependințe
```bash
pip install -r requirements.txt
```

### Antrenare (reproduce toate modelele)
```bash
python train.py
```

### Inferență cu modelul optim
```python
import joblib
import numpy as np

# Încărcare model și preprocesor
model = joblib.load('models/model_xgb_tuned.joblib')
preprocessor = joblib.load('models/preprocessor.joblib')
thresholds = joblib.load('models/xgb_thresholds.joblib')

# Preprocesare date noi
X_new_processed = preprocessor.transform(X_new)

# Predicție cu threshold optimizat pentru F1
probabilities = model.predict_proba(X_new_processed)[:, 1]
predictions = (probabilities >= thresholds['best_f1']).astype(int)
```

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
