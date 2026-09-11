# FraudShield AI — Setup & Run Guide

Two ways to run this: **VS Code** (full experience — API, Docker, tests, everything)
or **Google Colab** (fastest way to just see the ML pipeline run, no API/Docker).

---

## Option A: VS Code (recommended — full project)

### Step 1 — Install prerequisites
You need **Python 3.11 or 3.12** and **VS Code** installed on your machine.
Check your Python version first:
```bash
python3 --version
```
If you don't have Python 3.11+, install it from https://www.python.org/downloads/
(Windows/Mac) or `sudo apt install python3.12` (Ubuntu/Debian).

Install VS Code from https://code.visualstudio.com/ if you don't have it.
Inside VS Code, install the **Python extension** (Extensions icon on the left
sidebar → search "Python" → install the Microsoft one).

### Step 2 — Unzip the project
1. Download `fraudshield-ai.zip` (from this chat).
2. Unzip it anywhere on your computer, e.g. `Desktop/fraudshield-ai/`.
3. In VS Code: **File → Open Folder** → select the unzipped `fraudshield-ai` folder.

### Step 3 — Add the dataset
1. Download `creditcard.csv` (from this chat).
2. Inside the project folder, create a folder named `data/raw` if it doesn't
   already exist.
3. Put `creditcard.csv` inside it, so the path is:
   `fraudshield-ai/data/raw/creditcard.csv`

   (Alternative: skip this and instead run `make data` in Step 5 — it
   re-downloads the same file automatically.)

### Step 4 — Open a terminal and create a virtual environment
In VS Code: **Terminal → New Terminal** (top menu), then run:
```bash
python3 -m venv .venv
```
Activate it:
- **Mac/Linux:** `source .venv/bin/activate`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (cmd.exe):** `.venv\Scripts\activate.bat`

You'll know it worked because your terminal prompt now starts with `(.venv)`.
In VS Code, also click the Python version in the bottom-right status bar and
select the `.venv` interpreter, so VS Code itself (not just the terminal)
uses the right environment.

### Step 5 — Install dependencies
```bash
pip install -r requirements.txt
```
This installs pandas, scikit-learn, xgboost, shap, mlflow, optuna, fastapi,
uvicorn, pytest, etc. Takes a few minutes.

### Step 6 — Run the pipeline, step by step
Run these one at a time from the project root (each one prints real results
to the terminal and saves a `_MEASURED.json`/`.csv` file into `docs/`):

```bash
# 1. Validate the raw data (schema, nulls, duplicates)
python3 -m src.fraudshield.data.validate

# 2. Split into train/val/test (temporal split)
python3 src/fraudshield/data/split.py

# 3. Train & benchmark baseline models (Logistic Regression, Random Forest, XGBoost)
python3 training/train_baselines.py

# 4. Hyperparameter search (Optuna, 25 trials — takes a couple of minutes)
python3 training/optimize.py

# 5. Pick the serving threshold based on the business objective
python3 training/optimize_threshold.py

# 6. Final evaluation on the untouched test set
python3 training/evaluate.py

# 7. SHAP explainability
python3 src/fraudshield/explainability/shap_explainer.py

# 8. Drift monitoring report
python3 src/fraudshield/monitoring/drift.py

# 9. Register the best model in the MLflow registry
python3 training/register_model.py

# 10. Run all tests
python3 -m pytest tests/ -v
```
Or, once the dataset is in `data/raw/`, just run everything with one command:
```bash
make baselines optimize threshold evaluate shap drift test
```
(`make` comes pre-installed on Mac/Linux. On Windows, either install `make`
via `choco install make`, or just run the numbered `python3` commands above
one at a time instead — they do exactly what the Makefile targets do.)

### Step 7 — Start the API and try it
```bash
uvicorn src.fraudshield.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Then open your browser to **http://localhost:8000/docs** — this gives you
an interactive Swagger UI where you can click "Try it out" on `/predict`,
`/health`, `/model/info`, etc. and send real requests without writing any code.

To stop the server, go back to the terminal and press `Ctrl+C`.

### Step 8 (optional) — MLflow experiment dashboard
In a **new** terminal (keep the API running in the first one), with the venv
activated:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
Open **http://localhost:5001** to browse every training run, its parameters,
and its metrics.

### Step 9 (optional) — Docker
Only if you have Docker Desktop installed (https://www.docker.com/products/docker-desktop/):
```bash
docker compose build
docker compose up -d
```
API will be reachable at `http://localhost:8080` (via nginx) or
`http://localhost:8000` directly. `docker compose down` to stop.
**Note:** this was written to a production standard but not execution-tested
in the environment that built it (no Docker available there) — if something
doesn't build cleanly, that's expected to debug, not a guaranteed drop-in.

---

## Option B: Google Colab (fastest — pipeline only, no API/Docker)

Colab is a hosted Jupyter notebook — good for quickly seeing the ML pipeline
(data → models → SHAP) run, but it can't run a persistent API server or
Docker, so skip straight to a notebook-style flow.

### Step 1 — Open Colab
Go to https://colab.research.google.com/ → **File → New notebook**.

### Step 2 — Upload the project files
In the left sidebar, click the folder icon → the upload icon (page with an
up-arrow). Upload:
- `creditcard.csv`
- The whole `fraudshield-ai` folder is easiest to get in via Google Drive
  (see Step 2b) rather than uploading each file one by one through the
  browser.

**Step 2b (recommended) — use Google Drive instead of manual upload:**
1. Upload the unzipped `fraudshield-ai` folder to your Google Drive (e.g.
   `My Drive/fraudshield-ai/`), and put `creditcard.csv` inside
   `fraudshield-ai/data/raw/`.
2. In your Colab notebook, run:
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/fraudshield-ai
```

### Step 3 — Install dependencies
In a new cell:
```python
!pip install -q xgboost shap mlflow optuna fastapi uvicorn scikit-learn pandas numpy
```

### Step 4 — Run the pipeline
Colab runs shell commands if you prefix them with `!`. In separate cells:
```python
!python3 -m src.fraudshield.data.validate
```
```python
!python3 src/fraudshield/data/split.py
```
```python
!python3 training/train_baselines.py
```
```python
!python3 training/optimize.py
```
```python
!python3 training/optimize_threshold.py
```
```python
!python3 training/evaluate.py
```
```python
!python3 src/fraudshield/explainability/shap_explainer.py
```
```python
!python3 src/fraudshield/monitoring/drift.py
```
```python
!python3 -m pytest tests/unit tests/integration -v
```
Run each cell one at a time (Shift+Enter) and read the printed output — it's
the same real, measured numbers described in `docs/experiments.md`.

### Step 5 — Try the API in Colab (optional, a bit more setup)
Colab can run a server, you just need a tunnel to reach it from your browser:
```python
!pip install -q pyngrok
!python3 -m uvicorn src.fraudshield.api.main:app --host 0.0.0.0 --port 8000 &
```
```python
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(public_url)
```
Open the printed URL + `/docs` in your browser (you'll need a free ngrok
account/token the first time — https://ngrok.com/ — Colab will prompt you).
This is genuinely more hassle than just running it locally in VS Code
(Option A, Step 7) — Colab is really best suited to Steps 1–4 above.

### Step 6 — View results
Every script writes its output into `docs/*.json` / `docs/*.csv` inside the
project folder — view them directly in Colab:
```python
import json
print(json.dumps(json.load(open("docs/final_test_evaluation_MEASURED.json")), indent=2))
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run commands from the **project root** folder (the one containing `src/`, `training/`, `data/`), not from inside a subfolder. |
| `FileNotFoundError: data/raw/creditcard.csv` | You skipped Step 3 (VS Code) / Step 2b (Colab) — the CSV needs to be at exactly `data/raw/creditcard.csv` relative to where you run the command. |
| `pip install` fails on `xgboost`/`shap` | Make sure you're using Python 3.11 or 3.12 (not 3.13+ or 2.x) — run `python3 --version` to check. |
| Port 8000 already in use | Either stop whatever's using it, or run uvicorn with `--port 8001` instead and use that port in the browser URL. |
| `make: command not found` (Windows) | Run the individual `python3 ...` commands from Step 6 directly instead of `make`. |
