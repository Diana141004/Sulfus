# 📋 CHEATSHEET COMPLET — Arhitectură, Decizii, Parametri & MCP

> Documentație completă cap-coadă pentru prezentare/întrebări la internship.

---

## 🏗️ ARHITECTURA GENERALĂ

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  telecomunicatii│     │                  │     │   models/           │
│  .csv           │────▶│   train.py       │────▶│   ├── model_xgb_   │
│  (3000 rows)    │     │  (antrenare)     │     │   │   tuned.joblib  │
└─────────────────┘     └──────────────────┘     │   ├── preprocessor  │
                                                  │   │   .joblib       │
                                                  │   ├── xgb_thresholds│
                                                  │   │   .joblib       │
                                                  │   └── ...           │
                                                  └────────┬────────────┘
                                                           │
                               ┌────────────────────────────┤
                               │                            │
                    ┌──────────▼──────────┐     ┌───────────▼──────────┐
                    │   main.py           │     │   mcp_server.py      │
                    │   (FastAPI)         │     │   (FastMCP)          │
                    │   Port: 8000        │     │   stdio / SSE:8001   │
                    │   POST /predict     │     │   Tool: predict_     │
                    └─────────────────────┘     │   payment_delay      │
                                                └──────────────────────┘
```

### Flow complet:
1. `train.py` → citește CSV → antrenează modele → salvează artefacte în `models/`
2. `main.py` (FastAPI) → încarcă model + preprocessor + thresholds → expune REST API
3. `mcp_server.py` (FastMCP) → încarcă aceleași artefacte → expune ca MCP Tool

---

## 📦 ARTEFACTE SALVATE (models/)

| Fișier | Ce conține | Folosit de |
|--------|-----------|------------|
| `model_xgb_tuned.joblib` | Modelul XGBoost antrenat (best params) | main.py, mcp_server.py |
| `preprocessor.joblib` | ColumnTransformer (StandardScaler + OneHotEncoder) | main.py, mcp_server.py |
| `xgb_thresholds.joblib` | Dict: `{'best_f1': 0.756, 'high_recall': 0.473}` | main.py, mcp_server.py |
| `rezultate_comparare.json` | Metrici ale tuturor modelelor | mcp_server.py (resource) |

---

## 🔬 TRAIN.PY — Pipeline de Antrenare

### Decizia 1: Split-ul datelor
```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
- **80% train / 20% test**
- `stratify=y` → menține proporția 86%/14% identic în ambele seturi
- `random_state=42` → reproducibilitate

### Decizia 2: Preprocessing (ColumnTransformer)
```python
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),      # 11 coloane numerice
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # 4 coloane → 69 total
])
```

**De ce ColumnTransformer:**
- Aplică transformări diferite pe coloane diferite într-un singur obiect
- Se salvează ca un singur fișier (`preprocessor.joblib`)
- La inferență, `preprocessor.transform(df_input)` face EXACT aceeași transformare

**Features numerice (11):**
- `account_length`, `number_vmail_messages`, `total_day_minutes`, `total_day_calls`
- `total_eve_minutes`, `total_eve_calls`, `total_night_minutes`, `total_night_calls`
- `total_intl_minutes`, `total_intl_calls`, `number_customer_service_calls`

**Features categoriale (4) → expandate prin OneHot:**
- `state` → 51 coloane (50 state-uri + DC)
- `area_code` → 3 coloane (408, 415, 510)
- `international_plan` → 2 coloane (yes, no)
- `voice_mail_plan` → 2 coloane (yes, no)

**Total features după preprocessing: 69**

### Decizia 3: Eliminare charge columns
```python
cols_to_drop = ['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge']
```
**De ce:** `charge = minutes × rate_fix` → corelație perfectă (multicoliniaritate). Nu adaugă informație nouă, dar poate confuza modelul.

### Decizia 4: Handling dezechilibru clase (86% vs 14%)
```python
scale_pos_weight = neg / pos  # ≈ 6.26
```
- XGBoost: `scale_pos_weight` → dă greutate mai mare clasei minoritare
- Logistic Regression / SVM / RF: `class_weight='balanced'`
- CNN: `class_weight = {0: weight_for_0, 1: weight_for_1}`

### Decizia 5: Fine-tuning XGBoost (RandomizedSearchCV)

```python
param_distributions = {
    'n_estimators': [100, 200, 300, 500, 700],     # Câți arbori
    'max_depth': [3, 4, 5, 6, 7, 8],               # Adâncimea fiecărui arbore
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15], # Pas de învățare
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],        # % din date per arbore
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0], # % din features per arbore
    'min_child_weight': [1, 3, 5, 7],               # Min instanțe per leaf
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],         # L1 regularization
    'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0],       # L2 regularization
    'gamma': [0, 0.1, 0.3, 0.5, 1.0],              # Min loss reduction per split
    'scale_pos_weight': [spw, spw*1.2, spw*1.5],   # Weight for positive class
}
```

**Configurare search:**
- `n_iter=80` → 80 combinații random (nu toate 5^10 = ~10M)
- `scoring='recall'` → optimizăm pentru a prinde cât mai mulți clienți cu risc
- `cv = StratifiedKFold(n_splits=5)` → 5-fold cross-validation stratificat
- `n_jobs=-1` → paralelizare pe toate core-urile

**Best params găsite:**
```python
{'subsample': 0.6, 'scale_pos_weight': 9.41, 'reg_lambda': 0.5, 
 'reg_alpha': 1.0, 'n_estimators': 200, 'min_child_weight': 5, 
 'max_depth': 3, 'learning_rate': 0.03, 'gamma': 0, 'colsample_bytree': 0.8}
```

### Decizia 6: Threshold Tuning pe OOF (FIX-ul major!)

#### ❌ CE ERA ÎNAINTE (GREȘIT — data leakage):
```python
# Threshold calculat PE TEST SET
y_prob_tuned = xgb_tuned.predict_proba(X_test_processed)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_tuned)
best_threshold_f1 = thresholds[np.argmax(f1_scores)]  # 0.768

# Evaluare PE ACELAȘI TEST SET → metrici optimiste!
y_pred = (y_prob_tuned >= 0.768).astype(int)
f1_score(y_test, y_pred)  # 85.9% — biased!
```

**Problema:** Alegi threshold-ul care funcționează cel mai bine pe TEST, apoi raportezi performanța pe același TEST. E ca și cum ai da examenul, apoi alegi răspunsurile după ce vezi întrebările.

#### ✅ CE E ACUM (CORECT — Out-of-Fold):
```python
# Pas 1: Colectăm probabilități OOF (fiecare sample e prezis când NU era în train)
oof_probs = np.zeros(len(y_train))
for train_idx, val_idx in cv.split(X_train_processed, y_train_arr):
    fold_model = XGBClassifier(**best_params)
    fold_model.fit(X_train_processed[train_idx], y_train_arr[train_idx])
    oof_probs[val_idx] = fold_model.predict_proba(X_train_processed[val_idx])[:, 1]

# Pas 2: Threshold tuning pe OOF (nu pe test!)
precisions_oof, recalls_oof, thresholds_oof = precision_recall_curve(y_train_arr, oof_probs)
best_threshold_f1 = thresholds_oof[np.argmax(f1_scores_oof)]  # 0.756

# Pas 3: Model final antrenat pe TOT train-ul
xgb_tuned = XGBClassifier(**best_params)
xgb_tuned.fit(X_train_processed, y_train_arr)

# Pas 4: Evaluare pe TEST (complet nebiased!)
y_pred = (xgb_tuned.predict_proba(X_test_processed)[:, 1] >= 0.756).astype(int)
f1_score(y_test, y_pred)  # 85.35% — real, nebiased
```

**De ce e corect:** Fiecare sample din OOF e prezis de un model care NU l-a văzut la antrenare. Deci threshold-ul e ales pe date "nevăzute" — similar cu ce va fi pe test, dar fără a atinge test-ul.

---

## 🌐 MAIN.PY — FastAPI Server

### Ce e FastAPI?
- Framework Python pentru API-uri web
- Generează automat documentație Swagger (la `/docs`)
- Validare automată cu Pydantic
- Async, rapid

### Structura:

```python
# 1. Încărcare artefacte la startup
model = joblib.load("models/model_xgb_tuned.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")
thresholds = joblib.load("models/xgb_thresholds.joblib")

# 2. Schema de input (Pydantic validează automat)
class InputData(BaseModel):
    state: str
    account_length: int
    # ... toate câmpurile

# 3. Endpoint POST /predict
@app.post("/predict")
async def face_predictie(data: InputData, threshold_mode: str = "best_f1"):
    df_input = pd.DataFrame([data.model_dump()])
    df_input.drop(columns=charge_cols)    # Elimină charge
    X_processed = preprocessor.transform(df_input)  # CORECT!
    prob = model.predict_proba(X_processed)[:, 1][0]
    prediction = int(prob >= thresholds[threshold_mode])
    return {...}
```

### ❌ CE ERA ÎNAINTE (GREȘIT):
```python
# MODEL_COLUMNS = lista de 69 coloane hardcodate
df_input = pd.get_dummies(df_input, columns=categorical_features)
df_input = df_input.reindex(columns=MODEL_COLUMNS, fill_value=0)
# ❌ FĂRĂ StandardScaler! Datele numerice ajung ne-scalate la model
# ❌ MODEL_COLUMNS hardcodat — fragil, trebuie actualizat manual
```

**Problema:** Modelul a fost antrenat pe date scalate (StandardScaler), dar la inferență primea date brute. Predicțiile erau INCORECTE.

### ✅ CE E ACUM (CORECT):
```python
X_processed = preprocessor.transform(df_input)
# ✅ StandardScaler + OneHotEncoder — identic cu antrenarea
# ✅ Nu mai avem MODEL_COLUMNS hardcodat
# ✅ handle_unknown='ignore' → funcționează chiar dacă vine un state necunoscut
```

### Endpoints:

| Endpoint | Metoda | Ce face |
|----------|--------|---------|
| `/` | GET | Health check |
| `/health` | GET | Status + thresholds încărcate |
| `/predict` | POST | Predicție payment delay |
| `/docs` | GET | Swagger UI (auto-generat) |

### Cum rulezi:
```bash
uvicorn main:app --reload --port 8000
# Apoi: http://localhost:8000/docs
```

---

## 🤖 MCP SERVER — Explicație Completă

### Ce e MCP (Model Context Protocol)?

**MCP** = un protocol standardizat care permite unui LLM (Claude, GPT) să apeleze **tool-uri externe** în timp real.

Gândește-te la asta:
- Fără MCP: LLM-ul poate doar genera text
- Cu MCP: LLM-ul poate apela funcții, citi date, interacționa cu sisteme externe

### Analogie:
```
LLM (Claude) ←──MCP Protocol──→ MCP Server (mcp_server.py) ──→ Model ML
     │                                    │
     │  "Analizează clientul X"           │  predict_payment_delay(...)
     │  ────────────────────────────▶     │  ──────────────▶ XGBoost
     │                                    │
     │  "Clientul NU riscă plata"         │  {prediction: 0, prob: 0.18}
     │  ◀────────────────────────────     │  ◀──────────────
```

### Componentele MCP:

| Componentă | Ce e | Exemplu din proiect |
|-----------|------|---------------------|
| **Tool** | Funcție apelabilă de LLM | `predict_payment_delay(...)` |
| **Resource** | Date consultabile | `stats://dataset`, `stats://model-comparison` |
| **Prompt** | Template de conversație | `interpret_prediction`, `analyze_customer` |

### Transporturi (cum comunică):

| Transport | Comandă | Când se folosește |
|-----------|---------|-------------------|
| **stdio** | `python mcp_server.py` | Claude Desktop, Cline (local) |
| **SSE** | `fastmcp run mcp_server.py --transport sse --port 8001` | OpenWebUI, remote |
| **Inspector** | `fastmcp dev mcp_server.py` | Testing/debug (UI web) |

### Ce e stdio?
- Comunicare prin stdin/stdout (pipe-uri)
- Aplicația-gazdă (Claude Desktop) pornește procesul Python
- Trimite cereri JSON pe stdin, primește răspunsuri pe stdout
- **Nu are port, nu e web**

### Ce e SSE (Server-Sent Events)?
- Server HTTP care menține o conexiune deschisă
- Clientul se conectează la `http://localhost:8001/sse`
- Serverul trimite evenimente când sunt disponibile
- **Are port, e web, accesibil pe rețea**

### Ce e MCP Inspector?
- Tool de debugging — pornește un UI web
- Poți testa tool-urile manual din browser
- `fastmcp dev mcp_server.py` → deschide browser

### Structura mcp_server.py:

```python
from fastmcp import FastMCP

# 1. Inițializare server
mcp = FastMCP("Telecom Payment Delay Predictor", instructions="...")

# 2. Încărcare artefacte
model = joblib.load("models/model_xgb_tuned.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")
thresholds = joblib.load("models/xgb_thresholds.joblib")

# 3. Definire TOOL (funcție apelabilă de LLM)
@mcp.tool()
def predict_payment_delay(state: str, account_length: int, ...) -> dict:
    """Docstring = descrierea pe care LLM-ul o vede"""
    df_input = pd.DataFrame([data])
    X_processed = preprocessor.transform(df_input)
    prob = model.predict_proba(X_processed)[:, 1][0]
    return {"prediction": int(prob >= threshold), ...}

# 4. Definire RESOURCE (date consultabile)
@mcp.resource("stats://dataset")
def get_dataset_stats() -> str:
    """LLM-ul poate citi statistici despre dataset"""
    return "Total clienți: 3000, Payment delay YES: 13.8%..."

# 5. Definire PROMPT (template de conversație)
@mcp.prompt()
def interpret_prediction(probability: str, prediction: str, customer_info: str) -> str:
    """Template structurat pentru LLM"""
    return f"Analizează rezultatul: {prediction}..."

# 6. Pornire
if __name__ == "__main__":
    mcp.run()  # Default: stdio
```

### FastMCP vs MCP SDK:

| | FastMCP | MCP SDK (oficial) |
|--|---------|-------------------|
| Complexitate | Simplu (decoratori) | Verbose (clase, handlers) |
| Cod necesar | ~50 linii | ~200 linii |
| Syntax | `@mcp.tool()` | Manual registration |
| Documentare tool | Docstring automat | Manual |

### Comenzi FastMCP:

```bash
# Listare tool-uri disponibile
fastmcp list mcp_server.py

# Inspector web (debug)
fastmcp dev mcp_server.py

# Pornire server SSE (web)
fastmcp run mcp_server.py --transport sse --port 8001

# Pornire server stdio (pentru Claude Desktop)
python mcp_server.py
```

### Cum se configurează în Claude Desktop (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "telecom-predictor": {
      "command": "python",
      "args": ["C:/Users/Cristi/Desktop/Sesiune Fizica Luni/mcp_server.py"]
    }
  }
}
```

---

## 📊 PARAMETRII XGBoost — Explicați

| Parametru | Valoare | Ce face |
|-----------|---------|---------|
| `n_estimators=200` | Numărul de arbori din ensemble | Mai mulți = mai bun dar mai lent |
| `max_depth=3` | Adâncimea max a fiecărui arbore | Mic = regularizare, previne overfitting |
| `learning_rate=0.03` | Cât "învață" din fiecare arbore | Mic = convergență lentă dar stabilă |
| `subsample=0.6` | 60% din date per arbore | Bagging — reduce overfitting |
| `colsample_bytree=0.8` | 80% din features per arbore | Feature bagging |
| `min_child_weight=5` | Min 5 instanțe per leaf | Previne split-uri pe noise |
| `reg_alpha=1.0` | L1 regularization | Sparsity — elimină features neimportante |
| `reg_lambda=0.5` | L2 regularization | Reduce magnitude weights |
| `gamma=0` | Min loss reduction per split | 0 = permite orice split profitabil |
| `scale_pos_weight=9.41` | Weight clasă pozitivă | Compensează dezechilibrul 86%/14% |
| `eval_metric='logloss'` | Metric de evaluare internă | Log-loss pentru clasificare binară |

### De ce XGBoost > Random Forest > Logistic Regression?

| Aspect | LR | RF | XGBoost |
|--------|----|----|---------|
| Non-linearitate | ❌ | ✅ | ✅ |
| Feature interactions | ❌ | ✅ | ✅ |
| Regularizare | L1/L2 simplu | Bagging | L1+L2+gamma+depth |
| Handling dezechilibru | class_weight | class_weight | scale_pos_weight + tuning |
| Gradient boosting | ❌ | ❌ | ✅ (învață din erori) |

---

## 🔄 COMPARAȚIE: CE ERA vs CE E ACUM

### Threshold:
| | Înainte (v1.0) | Acum (v2.0) |
|--|-----------------|-------------|
| Calculat pe | Test set ❌ | Out-of-Fold (train) ✅ |
| Valoare | 0.768 | 0.756 |
| F1 raportat | 85.9% (biased) | 85.35% (real) |
| Data leakage | DA ❌ | NU ✅ |

### Preprocessing la inferență:
| | Înainte (v1.0) | Acum (v2.0) |
|--|-----------------|-------------|
| Metoda | `pd.get_dummies()` manual | `preprocessor.transform()` |
| StandardScaler | ❌ LIPSEA | ✅ Aplicat |
| Coloane | 69 hardcodate în `MODEL_COLUMNS` | Automat din preprocessor |
| State necunoscut | Crash/eroare | `handle_unknown='ignore'` → 0 |
| Consistență | ❌ Diferit de train | ✅ Identic cu train |

### Threshold la inferență:
| | Înainte (v1.0) | Acum (v2.0) |
|--|-----------------|-------------|
| Stocare | `THRESHOLD = 0.768` hardcodat | `joblib.load('xgb_thresholds.joblib')` |
| Schimbare | Editezi codul | Re-rulezi train.py, se actualizează automat |
| Multiple moduri | ❌ doar F1 | ✅ `best_f1` și `high_recall` |

---

## 🎯 METRICI — Ce înseamnă fiecare

| Metrică | Formula | Ce ne spune | Valoare noastră |
|---------|---------|-------------|-----------------|
| **AUC** | Area Under ROC Curve | Capacitatea generală de discriminare | 92.77% |
| **F1** | 2×(P×R)/(P+R) | Echilibru între Precision și Recall | 85.35% |
| **Recall** | TP/(TP+FN) | Câți clienți cu risc am prins | 80.72% |
| **Precision** | TP/(TP+FP) | Dintre cei alertați, câți chiar aveau risc | 90.54% |

### Trade-off Threshold:
- **Threshold MIC (0.473)** → Recall mare (87.95%) dar Precision mică (53.28%)
  - Prinzi mulți clienți cu risc, dar ai multe alarme false
- **Threshold MARE (0.756)** → Precision mare (90.54%) dar Recall mai mic (80.72%)
  - Când alertezi, aproape sigur ai dreptate, dar ratezi câțiva

---

## 🧪 TESTARE RAPIDĂ

### Test FastAPI:
```bash
uvicorn main:app --port 8000
# Apoi browser: http://localhost:8000/docs
# Sau curl:
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"state\":\"OH\",\"account_length\":107,\"area_code\":\"area_code_415\",\"international_plan\":\"no\",\"voice_mail_plan\":\"yes\",\"number_vmail_messages\":26,\"total_day_minutes\":161.6,\"total_day_calls\":123,\"total_day_charge\":27.47,\"total_eve_minutes\":195.5,\"total_eve_calls\":103,\"total_eve_charge\":16.62,\"total_night_minutes\":254.4,\"total_night_calls\":103,\"total_night_charge\":11.45,\"total_intl_minutes\":13.7,\"total_intl_calls\":3,\"total_intl_charge\":3.7,\"number_customer_service_calls\":1}"
```

### Test MCP:
```bash
# Listare
fastmcp list mcp_server.py

# Inspector web
fastmcp dev mcp_server.py

# SSE server
fastmcp run mcp_server.py --transport sse --port 8001
```

### Re-antrenare:
```bash
python train.py
# Durează ~2-3 min, salvează totul în models/
```

---

## 📚 LIBRĂRII CHEIE

| Librărie | Versiune | Rol |
|----------|----------|-----|
| `xgboost` | 3.2.0 | Modelul principal (gradient boosting) |
| `scikit-learn` | 1.8.0 | Preprocessing, metrici, alte modele |
| `fastapi` | 0.136.3 | REST API server |
| `uvicorn` | 0.48.0 | ASGI server (rulează FastAPI) |
| `fastmcp` | 3.3.1 | Framework MCP (Model Context Protocol) |
| `pydantic` | 2.13.4 | Validare date input (schemas) |
| `joblib` | 1.5.3 | Serializare modele ML pe disk |
| `catboost` | 1.2.10 | Model alternativ (gradient boosting) |
| `tensorflow` | 2.21.0 | CNN 1D (deep learning) |
| `pandas` | 3.0.3 | Manipulare date tabulare |
| `numpy` | 2.4.6 | Operații numerice |

---

## ❓ ÎNTREBĂRI POSIBILE & RĂSPUNSURI

**Q: De ce XGBoost și nu Random Forest?**
A: XGBoost face gradient boosting (învață din erorile arborilor anteriori), are regularizare built-in (L1, L2, gamma), și suportă nativ scale_pos_weight pentru clase dezechilibrate. RF doar face averaging pasiv.

**Q: Ce e data leakage?**
A: Când informație din test set "se scurge" în procesul de antrenare/tuning. La noi: alegeam threshold-ul pe test, apoi evaluam pe același test → metrici artificial bune.

**Q: De ce OOF și nu un validation set separat?**
A: Cu 3000 de rânduri și doar 413 pozitive, un validation set de 10% ar avea ~40 pozitive — prea puțin pentru o curbă Precision-Recall stabilă. OOF folosește tot train-ul eficient.

**Q: De ce StandardScaler pe numerice?**
A: XGBoost nu are nevoie strict de scalare (e tree-based), dar SVM și LR au nevoie. Folosim același preprocessor pentru toate modelele, plus că scalarea ajută stabilitatea numerică.

**Q: Ce e threshold-ul și de ce nu e 0.5?**
A: Default 0.5 presupune clase echilibrate. La noi 86%/14% → modelul tinde să prezică "no". Coborând threshold-ul, permitem modelului să fie mai "sensibil" la clasa minoritară.

**Q: Cum funcționează MCP în practică?**
A: Claude Desktop pornește `python mcp_server.py` ca subprocess. Când user-ul întreabă ceva legat de predicții, Claude vede tool-ul disponibil, construiește parametrii din conversație, apelează funcția, și prezintă rezultatul.

**Q: Ce face `handle_unknown='ignore'`?**
A: Dacă vine un state care nu exista în train (ex: un typo), OneHotEncoder pune 0 pe toate coloanele acelui state în loc să crashuiască. Robustness în producție.

**Q: De ce salvezi preprocessor-ul separat?**
A: StandardScaler are media și std calculate pe train. Dacă le recalculezi pe date noi, obții valori diferite → predicții greșite. Salvarea asigură transformări identice.
