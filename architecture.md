# Architecture

## Component map

```
┌─────────────────────────────────────────────────────────────────────┐
│ OFFLINE / TRAINING                                                  │
│                                                                       │
│  data/raw/creditcard.csv                                            │
│         │  src/fraudshield/data/validate.py  (schema/null/dupe check)│
│         ▼                                                             │
│  src/fraudshield/data/split.py  (temporal train/val/test)            │
│         ▼                                                             │
│  src/fraudshield/features/pipeline.py  (fit on train only)           │
│         ▼                                                             │
│  training/train_baselines.py  (LR, RF, XGBoost — MLflow logged)      │
│         ▼                                                             │
│  training/optimize.py  (Optuna TPE search, 25 trials — MLflow logged)│
│         ▼                                                             │
│  training/optimize_threshold.py  (business-objective threshold)      │
│         ▼                                                             │
│  training/evaluate.py  (ONE-TIME test-set evaluation)                │
│         ▼                                                             │
│  training/register_model.py  (MLflow Model Registry, promotion gate) │
│         ▼                                                             │
│  data/features/models/  (model.joblib + feature_pipeline.joblib +    │
│                           model_config.json — one deployable unit)    │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ONLINE / SERVING                                                     │
│                                                                       │
│  Client                                                               │
│    │  HTTPS                                                          │
│    ▼                                                                 │
│  nginx (deployment/nginx.conf) — rate limiting, TLS termination point │
│    │                                                                  │
│    ▼                                                                 │
│  FastAPI (src/fraudshield/api/main.py)                               │
│    │  Pydantic validation                                            │
│    ▼                                                                 │
│  FraudInferencePipeline (src/fraudshield/models/inference_pipeline.py)│
│    │  feature_pipeline.transform()                                   │
│    ▼                                                                 │
│  XGBoost model.predict_proba()                                       │
│    │                                                                  │
│    ▼                                                                 │
│  RiskPolicy.decide() (src/fraudshield/models/risk_engine.py)         │
│    │  probability + threshold + business rule -> LOW/MEDIUM/HIGH     │
│    ▼                                                                 │
│  Response: {fraud_probability, risk_level, decision, model_version,  │
│             threshold_version, latency_ms}                            │
│                                                                       │
│  (optional) FraudExplainer.explain_row() -> SHAP top factors          │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ MONITORING                                                            │
│  src/fraudshield/monitoring/drift.py — PSI (features), KS (predictions)│
│  /metrics endpoint — request volume, decision mix, error count        │
│  Structured logs — request_id, model_version, decision, latency       │
└─────────────────────────────────────────────────────────────────────┘
```

## Why this shape

- **One serialized unit per model version** (`feature_pipeline.joblib` +
  model `.joblib` + `model_config.json`, all versioned together): guarantees
  the API can never apply the wrong feature transform to a given model, which
  is a common real-world source of silent train/serve skew.
- **Risk decision policy is a separate class from the model**
  (`RiskPolicy`), reading its thresholds from `model_config.json` rather than
  being hard-coded: risk appetite / review-queue capacity changes should not
  require retraining or redeploying the model artifact.
- **MLflow + a promotion gate script**, not just "save the best model":
  `training/register_model.py` only promotes a new model to Production if it
  beats the current Production model's validation PR-AUC — this is the
  Candidate → Validation → Production flow, enforced in code, not just
  described in a document.
- **nginx in front of FastAPI**: rate limiting and body-size limits belong at
  the edge, not duplicated into every endpoint handler.

## Scalability notes (see also `docs/system_design.md`)
- The FastAPI service is stateless per request (model loaded once at
  startup) — horizontally scalable by running more container replicas behind
  the same nginx/gateway.
- Batch scoring (`/predict/batch`) reuses the same in-process model, so
  throughput for batch requests scales with process count, not with request
  count — appropriate for a review-queue batch job, not a claim about
  distributed/streaming throughput (which this project does not implement).
- MLflow's sqlite backend is fine for a single-node/portfolio deployment;
  a real multi-writer production deployment would move to a Postgres/MySQL
  backend store, called out explicitly rather than silently assumed away.
