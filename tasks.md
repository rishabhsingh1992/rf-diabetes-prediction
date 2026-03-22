# Project Roadmap

## Immediate Priorities
- [ ] Refactor `main.py` into reusable functions (load data, preprocess, train, evaluate, export).
- [ ] Remove duplicated dataframe split statements and enforce a clean single training flow.
- [ ] Add robust path handling for dataset and model artifact locations using `pathlib`.
- [ ] Create API startup checks that fail fast when model/scaler files are missing.
- [ ] Standardize prediction response format as structured JSON (label, confidence, metadata).
- [ ] Add input value validation rules and domain constraints in the request schema.
- [ ] Add basic automated tests for model loading and `/predict` endpoint behavior.

## Future Prospects
- [ ] Add probability scores and threshold tuning for prediction decisions.
- [ ] Introduce model versioning and artifact metadata for reproducibility.
- [ ] Add experiment tracking (metrics, params, feature importance snapshots).
- [ ] Containerize the API for portable deployment.
- [ ] Add CI pipeline for linting, tests, and model artifact checks.
- [ ] Add optional batch prediction endpoint for multiple records.
- [ ] Implement monitoring hooks for request volume, latency, and prediction drift.

## Learning Objectives
- [ ] Master end-to-end ML workflow structuring for production-friendly code.
- [ ] Practice FastAPI validation patterns and response schema design.
- [ ] Learn robust artifact serialization and loading strategies with `joblib`.
- [ ] Strengthen test design for ML-backed APIs (unit, integration, contract tests).
- [ ] Deepen understanding of feature scaling impacts on tree-based workflows.
- [ ] Apply reproducibility patterns (seed management, environment pinning, metadata).
