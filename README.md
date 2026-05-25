# 📡 Predicție Payment Delay – Telecomunicații

Proiect de Machine Learning pentru predicția întârzierilor de plată ale clienților unei companii de telecomunicații, dezvoltat ca parte a procesului de selecție internship.

## 🎯 Obiectiv

Clasificare binară: **va întârzia clientul plata?** (`payment_delay`: yes/no)

---

## 📊 Rezultate (Partea 1 – Antrenare)

| Model                       | AUC        | F1         | Recall     | Precision  | Threshold |
| --------------------------- | ---------- | ---------- | ---------- | ---------- | --------- |
| Logistic Regression         | 81.59%     | 46.04%     | 73.49%     | 33.52%     | -         |
| SVM (RBF)                   | 90.54%     | 65.96%     | 74.70%     | 59.05%     | -         |
| Random Forest               | 93.20%     | 59.50%     | 43.37%     | 94.74%     | -         |
| XGBoost                     | 92.24%     | 83.23%     | 80.72%     | 85.90%     | -         |
| **XGBoost Tuned (best F1)** | **92.77%** | **85.35%** | **80.72%** | **90.54%** | **0.756** |
| XGBoost Tuned (high Recall) | 92.77%     | 66.36%     | 87.95%     | 53.28%     | 0.473     |
| CatBoost                    | 92.68%     | 83.13%     | 83.13%     | 83.13%     | -         |
| CNN 1D                      | 84.91%     | 42.02%     | 30.12%     | 69.44%     | -         |

**🏆 Best model: XGBoost Tuned** – optimizat cu RandomizedSearchCV (80 iterații, 5-fold CV) + threshold tuning pe **Out-of-Fold predictions** (fără data leakage).

---

## 🚀 Partea 2 – FastAPI (Inferență)

### Rulare server

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Endpoints

| Metoda | Endpoint   | Descriere                                 |
| ------ | ---------- | ----------------------------------------- |
| GET    | `/`        | Health check simplu                       |
| GET    | `/health`  | Status server + model loaded + thresholds |
| POST   | `/predict` | **Predicție payment delay**               |
| GET    | `/docs`    | Swagger UI (documentație interactivă)     |

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
    "metrics_on_test": {
      "f1": 0.8535,
      "precision": 0.9054,
      "recall": 0.8072,
      "auc": 0.9277
    }
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

| Tip      | Nume                       | Descriere                                            |
| -------- | -------------------------- | ---------------------------------------------------- |
| **Tool** | `predict_payment_delay`    | Predicție payment delay cu threshold optimizat (OOF) |
| Resource | `stats://dataset`          | Statistici generale despre dataset                   |
| Resource | `stats://model-comparison` | Tabel comparare toate modelele                       |
| Prompt   | `interpret_prediction`     | Template LLM pentru interpretare rezultat            |
| Prompt   | `analyze_customer`         | Template rapid analiză client                        |

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

## 🐳 Partea 4 – Docker & Docker Compose

Întregul proiect este containerizat pentru deployment facil și reproductibil. Arhitectura folosește **3 microservicii** orchestrate cu Docker Compose.

### Arhitectură

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Network                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  FastAPI      │  │  FastMCP     │  │    OpenWebUI       │  │
│  │  (Backend)    │  │  (MCP Server)│  │    (Frontend)      │  │
│  │  Port 8000    │  │  Port 8001   │  │    Port 3000       │  │
│  │              │  │              │  │                    │  │
│  │  uvicorn     │  │  SSE transport│  │  LLM Interface    │  │
│  │  main:app    │  │  mcp_server  │  │  + MCP client     │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
│         │                  │                   │              │
│         └──────────────────┴───────────────────┘              │
│                    Shared Docker Network                      │
└─────────────────────────────────────────────────────────────┘
```

### Dockerfile

```dockerfile
# Platformă universală (compatibilitate Mac M1/M2 + Linux)
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Layer caching: dependințe separate de cod
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir mcp requests fastmcp

# Copiere proiect (main.py, mcp_server.py, models/, etc.)
COPY . .

EXPOSE 8000 8001
```

**Decizii de design:**

- `--platform=linux/amd64` → compatibilitate cross-platform (Mac ARM → emulare amd64)
- `python:3.11-slim` → imagine minimală (~150MB vs ~1GB full)
- Două `RUN pip install` separate → layer caching eficient (dacă modifici doar pachetele MCP, nu reinstalezi tot)
- `--no-cache-dir` → imagine finală mai mică (nu stochează cache pip)

### Docker Compose (`docker-compose.yml`)

```yaml
services:
  # SERVICIUL 1: Backend REST API (FastAPI)
  telecom-api:
    platform: linux/amd64
    build: .
    container_name: ml_fastapi_backend
    command: uvicorn main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    restart: unless-stopped

  # SERVICIUL 2: Server AI (FastMCP via SSE)
  telecom-mcp:
    platform: linux/amd64
    build: .
    container_name: ml_mcp_server
    command: fastmcp run mcp_server.py --transport sse --port 8001
    ports:
      - "8001:8001"
    restart: unless-stopped

  # SERVICIUL 3: Interfață Web (OpenWebUI)
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open_webui_client
    ports:
      - "3000:8080"
    environment:
      - WEBUI_AUTH=false
      - AWS_ACCESS_KEY_ID=<your-key>
      - AWS_SECRET_ACCESS_KEY=<your-secret>
      - AWS_REGION=us-west-2
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - open-webui-data:/app/backend/data
    restart: unless-stopped

volumes:
  open-webui-data:
```

### Comenzi Docker

```bash
# Build și pornire toate serviciile
docker-compose up --build

# Pornire în background (detached)
docker-compose up --build -d

# Verificare status containere
docker-compose ps

# Vizualizare log-uri
docker-compose logs -f telecom-api      # log-uri FastAPI
docker-compose logs -f telecom-mcp      # log-uri MCP server
docker-compose logs -f open-webui       # log-uri OpenWebUI

# Oprire toate serviciile
docker-compose down

# Rebuild doar un serviciu specific
docker-compose up --build telecom-api
```

### Comunicare între servicii

Docker Compose creează automat o **rețea bridge** internă. Serviciile se pot accesa reciproc prin:

- `telecom-api:8000` — API REST (din alte containere)
- `telecom-mcp:8001` — MCP Server (din OpenWebUI)
- `host.docker.internal` — accesare servicii de pe mașina host

### Persistență date

| Volume            | Scop                                                                   |
| ----------------- | ---------------------------------------------------------------------- |
| `open-webui-data` | Configurări OpenWebUI, conversații, setări — persistă între restartări |

---

## 🌐 Partea 5 – OpenWebUI (Interfață AI)

[OpenWebUI](https://github.com/open-webui/open-webui) este o interfață web open-source (similară ChatGPT) care permite interacțiunea cu modele LLM și integrarea cu servere MCP.

### De ce OpenWebUI?

- ✅ **Open-source** și self-hosted (datele rămân locale)
- ✅ **Suport MCP nativ** — se conectează direct la serverul nostru FastMCP
- ✅ **Interfață intuitivă** — similar ChatGPT, ușor de folosit de non-tehnici
- ✅ **Containerizat** — o singură linie în docker-compose, zero configurare manuală
- ✅ **Multi-model** — suportă Claude (Bedrock), GPT, Ollama, etc.

### Configurare OpenWebUI cu MCP

După `docker-compose up`, accesați **http://localhost:3000** și:

1. **Conectare LLM Backend:**
   - Settings → Connections → adăugare provider Amazon Bedrock (Claude)
   - Credentials: AWS Access Key + Secret Key din environment variables

2. **Conectare MCP Server:**
   - Settings → Tools → MCP Servers → Add Server
   - URL: `http://telecom-mcp:8001/sse` (numele serviciului din docker-compose)
   - Serverul expune automat tool-ul `predict_payment_delay`

3. **Utilizare:**
   - Deschideți un chat nou
   - Întrebați: _"Analizează un client din Ohio cu 200 minute/zi și 4 apeluri la customer service. Va întârzia plata?"_
   - LLM-ul apelează automat tool-ul MCP și returnează predicția cu interpretare

### Flux complet (End-to-End)

```
Utilizator (browser)
    │
    ▼
OpenWebUI (localhost:3000)
    │
    ├──► LLM (Claude via AWS Bedrock) → generează text + decide să apeleze tool
    │
    ▼
MCP Server (telecom-mcp:8001)
    │
    ├──► Încarcă preprocessor.joblib
    ├──► Aplică StandardScaler + OneHotEncoder
    ├──► model_xgb_tuned.joblib.predict_proba()
    ├──► Compară cu threshold OOF (0.756)
    │
    ▼
Răspuns → LLM interpretează → Afișare utilizator
```

### Screenshot-uri workflow

1. **Utilizatorul pune o întrebare** în limbaj natural
2. **LLM-ul recunoaște** că are nevoie de predicția modelului ML
3. **Apelează tool-ul MCP** `predict_payment_delay` cu parametrii extrași
4. **Primește rezultatul** (predicție + probabilitate + threshold)
5. **Generează un răspuns** interpretat pentru utilizator (non-tehnic)

---

## 🛠️ Tehnologii

| Categorie          | Tehnologii                                        |
| ------------------ | ------------------------------------------------- |
| **ML/AI**          | scikit-learn, XGBoost, CatBoost, TensorFlow/Keras |
| **Backend**        | FastAPI, Uvicorn, Pydantic                        |
| **MCP**            | FastMCP (Model Context Protocol)                  |
| **Infrastructure** | Docker, Docker Compose                            |
| **Frontend**       | OpenWebUI                                         |
| **LLM**            | Claude (Amazon Bedrock)                           |
| **Data**           | Pandas, NumPy, joblib                             |
| **Embeddings**     | HuggingFace (all-MiniLM-L6-v2)                    |

## 📁 Structura proiectului

```
├── train.py                    # Partea 1: Pipeline complet de antrenare
├── main.py                     # Partea 2: FastAPI server pentru inferență
├── mcp_server.py               # Partea 3: Server FastMCP (Tool MCP)
├── Dockerfile                  # Partea 4: Containerizare aplicație
├── docker-compose.yml          # Partea 4: Orchestrare microservicii
├── requirements.txt            # Dependințe Python
├── telecomunicatii.csv         # Dataset (3000 clienți, 20 features)
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
8. **Containerizare**: Docker multi-service cu Docker Compose (FastAPI + MCP + OpenWebUI)
9. **Interfață AI**: OpenWebUI conectat la MCP Server pentru interacțiune în limbaj natural

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

| Mode          | Threshold | F1     | Recall | Precision | Când se folosește                                 |
| ------------- | --------- | ------ | ------ | --------- | ------------------------------------------------- |
| `best_f1`     | 0.756     | 85.35% | 80.72% | 90.54%    | Default — echilibru optim                         |
| `high_recall` | 0.473     | 66.36% | 87.95% | 53.28%    | Când vrei să prinzi cât mai mulți clienți cu risc |

---

## 🚀 Quick Start (Deploy complet)

```bash
# 1. Clonare repo
git clone https://github.com/Diana141004/Sulfus.git
cd Sulfus

# 2. Pornire toate serviciile (build + run)
docker-compose up --build -d

# 3. Verificare
docker-compose ps

# 4. Accesare interfețe:
#    - OpenWebUI:     http://localhost:3000
#    - FastAPI Docs:  http://localhost:8000/docs
#    - MCP Server:    http://localhost:8001

# 5. Oprire
docker-compose down
```

---

## 👥 Echipa

Proiect dezvoltat pentru procesul de selecție internship.
