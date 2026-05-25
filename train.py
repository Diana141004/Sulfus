"""
Pipeline de antrenare și comparare modele ML pentru predicția payment_delay
în telecomunicații.

Modele comparate: Logistic Regression, SVM, Random Forest, XGBoost (+ fine-tuned),
CatBoost, CNN 1D.

Cel mai bun model: XGBoost Tuned — threshold optimizat pe Out-of-Fold predictions
(fără data leakage pe test set).
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    make_scorer, precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Optional: TensorFlow/Keras (CNN)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Dense, Conv1D, GlobalAveragePooling1D,
        Dropout, BatchNormalization, Reshape
    )
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
except ImportError:
    HAS_TF = False

# ============================================================================
# THRESHOLD-URI OPTIME — se recalculează la fiecare antrenare prin OOF
# Valorile de mai jos sunt placeholder-uri actualizate la ultima rulare.
# La inferență, se încarcă din models/xgb_thresholds.joblib
# ============================================================================


def create_cnn_model(input_dim):
    """Arhitectura CNN 1D pentru clasificare binara."""
    if not HAS_TF:
        return None
    model = Sequential([
        Reshape((input_dim, 1), input_shape=(input_dim,)),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(model, X_test, y_test):
    """Evalueaza un model sklearn si returneaza metrici standard."""
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    return {
        'AUC': float(roc_auc_score(y_test, y_prob)),
        'F1': float(f1_score(y_test, y_pred)),
        'Recall': float(recall_score(y_test, y_pred)),
        'Precision': float(precision_score(y_test, y_pred))
    }


def main():
    # =========================================================================
    # 1. INCARCARE SI PREGATIRE DATE
    # =========================================================================
    print("===> 1. Incarcare si pregatire date...")
    if not os.path.exists('telecomunicatii.csv'):
        print("Eroare: Fisierul 'telecomunicatii.csv' nu exista in directorul curent.")
        return

    df = pd.read_csv('telecomunicatii.csv')

    if 'payment_delay' not in df.columns:
        print("Eroare: Coloana 'payment_delay' nu exista.")
        return

    df['payment_delay'] = df['payment_delay'].map({'yes': 1, 'no': 0})

    # Eliminare features redundante (multicolinearitate perfecta cu minutele)
    cols_to_drop = ['total_day_charge', 'total_eve_charge',
                    'total_night_charge', 'total_intl_charge']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    y = df['payment_delay']
    X = df.drop(columns=['payment_delay'])

    categorical_features = ['state', 'area_code', 'international_plan', 'voice_mail_plan']
    for c in categorical_features:
        if c in X.columns:
            X[c] = X[c].astype(str)

    numeric_features = [c for c in X.columns if c not in categorical_features]

    # Split stratificat 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

    # =========================================================================
    # 2. PREPROCESARE (One-Hot Encoding + Standard Scaling)
    # =========================================================================
    print("===> 2. Aplicare Transformari (One-Hot & Scalare)...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    print("    Preprocesor salvat: models/preprocessor.joblib")

    results = {}
    neg = sum(y_train == 0)
    pos = sum(y_train == 1)
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    # =========================================================================
    # 3. LOGISTIC REGRESSION (Baseline)
    # =========================================================================
    print("\n===> 3. Antrenare Logistic Regression (Baseline)...")
    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr.fit(X_train_processed, y_train)
    joblib.dump(lr, 'models/model_lr.joblib')
    results['Logistic Regression'] = evaluate_model(lr, X_test_processed, y_test)

    # =========================================================================
    # 4. SVM (RBF kernel)
    # =========================================================================
    print("===> 4. Antrenare SVM (RBF kernel)...")
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_train_processed, y_train)
    joblib.dump(svm, 'models/model_svm.joblib')
    results['SVM (RBF)'] = evaluate_model(svm, X_test_processed, y_test)

    # =========================================================================
    # 5. RANDOM FOREST
    # =========================================================================
    print("===> 5. Antrenare Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train_processed, y_train)
    joblib.dump(rf, 'models/model_rf.joblib')
    results['Random Forest'] = evaluate_model(rf, X_test_processed, y_test)

    # =========================================================================
    # 6. XGBOOST (default)
    # =========================================================================
    print("===> 6. Antrenare XGBoost (default)...")
    xgb_default = XGBClassifier(
        scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'
    )
    xgb_default.fit(X_train_processed, y_train)
    joblib.dump(xgb_default, 'models/model_xgb.joblib')
    results['XGBoost'] = evaluate_model(xgb_default, X_test_processed, y_test)

    # =========================================================================
    # 7. XGBOOST FINE-TUNED (RandomizedSearchCV + OOF Threshold Optimization)
    # =========================================================================
    print("===> 7. Fine-tuning XGBoost (RandomizedSearchCV + OOF Threshold Tuning)...")

    param_distributions = {
        'n_estimators': [100, 200, 300, 500, 700],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0],
        'gamma': [0, 0.1, 0.3, 0.5, 1.0],
        'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 1.2, scale_pos_weight * 1.5],
    }

    xgb_base = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        xgb_base,
        param_distributions=param_distributions,
        n_iter=80,
        scoring='recall',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X_train_processed, y_train)
    best_params = search.best_params_

    print(f"    Best CV Recall: {search.best_score_:.4f}")
    print(f"    Best params: {best_params}")

    # -------------------------------------------------------------------------
    # THRESHOLD TUNING PE OUT-OF-FOLD PREDICTIONS (fara data leakage!)
    # Antrenam modelul pe fiecare fold si colectam probabilitatile pe validare
    # -------------------------------------------------------------------------
    print("    Calculare threshold optim pe Out-of-Fold predictions...")
    oof_probs = np.zeros(len(y_train))
    y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_processed, y_train_arr)):
        fold_model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
        fold_model.fit(X_train_processed[train_idx], y_train_arr[train_idx])
        oof_probs[val_idx] = fold_model.predict_proba(X_train_processed[val_idx])[:, 1]

    # Threshold tuning pe curba Precision-Recall din OOF (nu din test!)
    precisions_oof, recalls_oof, thresholds_oof = precision_recall_curve(y_train_arr, oof_probs)

    # Threshold care maximizeaza F1 (pe OOF)
    f1_scores_oof = 2 * (precisions_oof * recalls_oof) / (precisions_oof + recalls_oof + 1e-8)
    best_f1_idx = np.argmax(f1_scores_oof)
    best_threshold_f1 = thresholds_oof[best_f1_idx] if best_f1_idx < len(thresholds_oof) else 0.5

    # Threshold care asigura Recall >= 85% cu F1 maxim (pe OOF)
    target_recall = 0.85
    valid_mask = recalls_oof[:-1] >= target_recall
    if valid_mask.any():
        candidates_f1 = f1_scores_oof[:-1][valid_mask]
        best_candidate_idx = np.where(valid_mask)[0][np.argmax(candidates_f1)]
        best_threshold_recall = thresholds_oof[best_candidate_idx]
    else:
        best_threshold_recall = best_threshold_f1

    print(f"    OOF Threshold best F1:     {best_threshold_f1:.4f}")
    print(f"    OOF Threshold high Recall: {best_threshold_recall:.4f}")

    # -------------------------------------------------------------------------
    # Antrenam modelul FINAL pe tot train-ul cu best_params
    # -------------------------------------------------------------------------
    xgb_tuned = XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
    xgb_tuned.fit(X_train_processed, y_train_arr)

    # Evaluare pe TEST cu threshold-urile gasite pe OOF (evaluare nebiased)
    y_prob_tuned = xgb_tuned.predict_proba(X_test_processed)[:, 1]

    y_pred_f1 = (y_prob_tuned >= best_threshold_f1).astype(int)
    results['XGBoost Tuned (best F1)'] = {
        'AUC': float(roc_auc_score(y_test, y_prob_tuned)),
        'F1': float(f1_score(y_test, y_pred_f1)),
        'Recall': float(recall_score(y_test, y_pred_f1)),
        'Precision': float(precision_score(y_test, y_pred_f1)),
        'Threshold': float(best_threshold_f1)
    }

    y_pred_recall = (y_prob_tuned >= best_threshold_recall).astype(int)
    results['XGBoost Tuned (high Recall)'] = {
        'AUC': float(roc_auc_score(y_test, y_prob_tuned)),
        'F1': float(f1_score(y_test, y_pred_recall)),
        'Recall': float(recall_score(y_test, y_pred_recall)),
        'Precision': float(precision_score(y_test, y_pred_recall)),
        'Threshold': float(best_threshold_recall)
    }

    print(f"    [TEST] Threshold best F1:     {best_threshold_f1:.4f} -> F1={f1_score(y_test, y_pred_f1):.4f}, Recall={recall_score(y_test, y_pred_f1):.4f}, Prec={precision_score(y_test, y_pred_f1):.4f}")
    print(f"    [TEST] Threshold high Recall: {best_threshold_recall:.4f} -> F1={f1_score(y_test, y_pred_recall):.4f}, Recall={recall_score(y_test, y_pred_recall):.4f}, Prec={precision_score(y_test, y_pred_recall):.4f}")

    # Salvare model si threshold-uri
    joblib.dump(xgb_tuned, 'models/model_xgb_tuned.joblib')
    joblib.dump(
        {'best_f1': float(best_threshold_f1), 'high_recall': float(best_threshold_recall)},
        'models/xgb_thresholds.joblib'
    )
    print("    Model salvat: models/model_xgb_tuned.joblib")
    print("    Threshold-uri salvate: models/xgb_thresholds.joblib")

    # =========================================================================
    # 8. CATBOOST
    # =========================================================================
    print("===> 8. Antrenare CatBoost...")
    cat_features_indices = [
        X_train.columns.get_loc(c) for c in categorical_features if c in X_train.columns
    ]
    catb = CatBoostClassifier(
        iterations=500, auto_class_weights='Balanced',
        random_state=42, verbose=0, cat_features=cat_features_indices
    )
    catb.fit(X_train, y_train)
    catb.save_model('models/model_catboost.cbm')

    y_pred_cb = catb.predict(X_test)
    y_pred_cb = np.array([
        1 if str(x).strip().lower() in ['1', '1.0', 'true', 'yes'] else 0
        for x in y_pred_cb
    ])

    try:
        y_prob_cb = catb.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob_cb = y_pred_cb

    results['CatBoost'] = {
        'AUC': float(roc_auc_score(y_test, y_prob_cb)),
        'F1': float(f1_score(y_test, y_pred_cb)),
        'Recall': float(recall_score(y_test, y_pred_cb)),
        'Precision': float(precision_score(y_test, y_pred_cb))
    }

    # =========================================================================
    # 9. CNN 1D (Deep Learning)
    # =========================================================================
    if HAS_TF:
        print("===> 9. Antrenare CNN 1D (Deep Learning)...")
        input_dim = X_train_processed.shape[1]
        cnn = create_cnn_model(input_dim)

        total = neg + pos
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        cnn.fit(
            X_train_processed, y_train,
            epochs=50, batch_size=32,
            validation_split=0.2, callbacks=[early_stopping],
            class_weight=class_weight, verbose=0
        )

        cnn.save('models/model_cnn.keras')

        y_prob_cnn = cnn.predict(X_test_processed, verbose=0).flatten()
        y_pred_cnn = (y_prob_cnn > 0.5).astype(int)

        results['CNN 1D'] = {
            'AUC': float(roc_auc_score(y_test, y_prob_cnn)),
            'F1': float(f1_score(y_test, y_pred_cnn)),
            'Recall': float(recall_score(y_test, y_pred_cnn)),
            'Precision': float(precision_score(y_test, y_pred_cnn))
        }
    else:
        print("===> 9. Sarit peste CNN (TensorFlow nu este instalat).")

    # =========================================================================
    # 10. SALVARE SI AFISARE REZULTATE
    # =========================================================================
    with open('models/rezultate_comparare.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 60)
    print("REZULTATE FINALE (Pe setul de test de 20%)")
    print("=" * 60)
    df_res = pd.DataFrame(results).T
    for col in df_res.columns:
        df_res[col] = (df_res[col] * 100).round(2).astype(str) + '%'
    print(df_res)
    print("=" * 60)
    print("\nToate modelele au fost salvate in folderul 'models/'.")
    print(f"\n*** BEST MODEL: XGBoost Tuned ***")
    print(f"    Threshold F1 optim (OOF):     {best_threshold_f1:.4f}")
    print(f"    Threshold Recall optim (OOF): {best_threshold_recall:.4f}")
    print("    (Threshold-uri calculate pe Out-of-Fold predictions, fara data leakage)")


if __name__ == "__main__":
    main()
