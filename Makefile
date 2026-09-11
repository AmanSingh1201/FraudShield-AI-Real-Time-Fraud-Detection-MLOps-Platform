.PHONY: install data split validate features baselines optimize threshold evaluate shap drift test lint api docker-build docker-up docker-down mlflow-ui all

install:
	pip install --break-system-packages -r requirements.txt

data:
	mkdir -p data/raw
	curl -sL -o data/raw/creditcard.csv \
	  https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv

validate:
	python3 -m src.fraudshield.data.validate

split:
	python3 src/fraudshield/data/split.py

baselines:
	python3 training/train_baselines.py

optimize:
	python3 training/optimize.py

threshold:
	python3 training/optimize_threshold.py

evaluate:
	python3 training/evaluate.py

shap:
	python3 src/fraudshield/explainability/shap_explainer.py

drift:
	python3 src/fraudshield/monitoring/drift.py

test:
	python3 -m pytest tests/ -v

lint:
	ruff check src/ training/ tests/

api:
	uvicorn src.fraudshield.api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# Full pipeline, in order — mirrors the phased build process in docs/experiments.md
all: data validate split baselines optimize threshold evaluate shap drift test
