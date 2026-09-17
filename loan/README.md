# Loan Default Risk Dashboard

This project contains a Streamlit dashboard for predicting loan default risk using a trained machine learning model.

## Project files

- `app.py` — Streamlit frontend
- `train_model.py` — training script to build the model and artifacts
- `Loan_default.csv` — dataset used for training
- model artifact files: `model.pkl`, `scaler.pkl`, `columns.pkl`, `threshold.pkl`, `metrics.pkl`, `feature_importance.pkl`

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

## Notes

- The app predicts default probability and shows a risk threshold gauge.
- It includes a model performance tab and a data overview tab.
- The threshold is tuned to maximize F1 score for imbalanced loan data.
