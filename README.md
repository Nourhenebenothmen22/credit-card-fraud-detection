# Credit Card Fraud Detection

End-to-end, reproducible machine learning project for credit card fraud detection:
imbalanced classification (SMOTE vs. alternatives), regularization, hyperparameter
optimization, threshold selection, SHAP explainability and a modular pipeline.

> Status: Phase 1 – project initialization completed.

## Requirements
- Ubuntu Linux
- Python 3.11

## Installation
```bash
git clone <repo-url> && cd credit-card-fraud-detection
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Project structure
- `data/` raw (immutable), interim and processed datasets (git-ignored)
- `notebooks/` exploratory analysis
- `src/` reusable, tested modules
- `tests/` unit and integration tests
- `configs/` experiment configuration
- `models/` persisted models (git-ignored)
- `reports/` figures, metrics and experiment logs
- `docs/` technical documentation

## Methodology rules
Split before preprocessing; fit transformations on training data only; SMOTE only
inside the training folds; test set used once for the final selected model.
