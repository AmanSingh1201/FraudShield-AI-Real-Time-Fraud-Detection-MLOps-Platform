# System Design

## Request-time path

```
Client → nginx (rate limit, body-size limit) → FastAPI (Pydantic validation)
       → FraudInferencePipeline (feature transform → XGBoost → RiskPolicy)
       → JSON response
```

- **Fault tolerance:** the API's startup hook loads the model once; if
  loading fails, `/health` returns 503 rather than the process crash-looping
  silently. All prediction exceptions are caught, logged with a request ID,
  and returned as a generic 500 — no stack trace or internal detail is ever
  returned to the client (verified in `tests/api/test_api.py::test_no_stack_trace_leaked_on_error`).
- **Batch vs. real-time:** `/predict` is for the synchronous authorization
  path (low latency required — measured p50 5.5ms / p99 9.2ms full HTTP
  round trip, see `docs/api_latency_benchmark_MEASURED.json`); `/predict/batch`
  is for offline review-queue scoring where per-call HTTP overhead matters
  less than throughput.
- **Model versioning at request time:** every response includes
  `model_version` and `threshold_version`, so a caller (or an audit log) can
  always tell which model made a given decision — required for the
  Candidate→Production rollback story below to be auditable after the fact.

## Training-time path

```
Data (data/raw/) → Validation → Temporal Split → Feature Pipeline (fit=train)
   → Baseline Models (MLflow) → Optuna Search (MLflow) → Threshold Optimization
   → Final Test Evaluation (touched once) → Model Registry (promotion gate)
   → Deployment artifact (data/features/models/)
   → Monitoring (drift) → Retraining trigger (docs/monitoring.md)
```

- **Reproducibility:** fixed `random_seed=42` throughout
  (`configs/data.yaml`, `configs/training.yaml`); every training script reads
  its config from `configs/`, not from ad hoc constants scattered in code
  (aside from the seed, which is intentionally centralized but repeated per
  script for clarity/testability); `requirements.txt` pins exact versions.
- **Versioning:** the deployable unit is
  `{feature_pipeline.joblib, model.joblib, model_config.json}` — all three
  are produced and updated together, so a model version can never be paired
  with the wrong feature transform.
- **Rollback:** MLflow Model Registry keeps archived (not deleted) prior
  Production versions; rollback is a registry stage transition, not a
  retrain.

## Scalability

- The service is stateless per-request (model held in memory, no per-request
  DB write on the hot path) — scales horizontally by adding container
  replicas behind nginx/a load balancer. This project ships a single-instance
  `docker-compose.yml`; a multi-replica deployment is a configuration change
  (more `fraudshield_api` replicas + a real load balancer), not an
  architecture change.
- Training is currently single-node (fits comfortably: 199K rows, XGBoost
  trains in 6.5s measured). At IEEE-CIS-like scale (~590K+ rows, richer
  categorical features) a distributed training step (e.g. Dask-XGBoost or
  Spark) would become worth the added operational complexity — not
  implemented here because the actual dataset used doesn't need it, and
  adding it without a real need would violate the project's
  don't-add-technology-for-technology's-sake principle.
- MLflow's `sqlite:///mlflow.db` tracking store is single-writer-friendly and
  fine at this scale; a team-scale deployment would move to a
  Postgres-backed MLflow server (the `docker-compose.yml` MLflow service is
  already factored out as its own container to make that swap a config
  change, not a redesign).

## Latency vs. batch trade-off
Real-time (`/predict`) optimizes for per-request latency at the cost of
per-request HTTP overhead; batch (`/predict/batch`) amortizes that overhead
across N transactions in one call at the cost of not being suitable for a
hard-real-time authorization decision. Both share the exact same
`FraudInferencePipeline.predict_one()` code path — there is one prediction
implementation, not two independently-maintained ones.

## What this design does NOT claim
- No Kubernetes: three containers (api, mlflow, nginx) via docker-compose is
  sufficient at this scale; adding an orchestrator would be technology for
  its own sake, per the project's explicit ground rule.
- No claim of a live production deployment — everything above was built,
  run, and measured in a local/sandbox environment (see each `_MEASURED.json`
  file for what was actually executed and when). Docker build/run itself
  could not be verified in this sandbox (no Docker daemon available here);
  the Dockerfile and compose file are written to the same
  non-root/multi-stage/healthcheck standard as the rest of the project but
  have not been execution-tested end-to-end. This is disclosed explicitly
  rather than claimed as verified.
- No claim of a live delayed-label feed or live drift dashboard — those are
  designed in `docs/monitoring.md` against the code that does exist
  (`drift.py`, `/metrics`), not implemented as a running service, because
  this is an offline Kaggle-style dataset with no live traffic to monitor.
