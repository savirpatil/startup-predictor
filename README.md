# Startup Outcome Probability Model

A logistic regression model and Streamlit app that estimates whether an early-stage
startup is more likely to be acquired/IPO or to close, based on funding history,
timing, industry, and global location.

Live app: https://startup-predictor-37xkf39vwzl762ads2sifh.streamlit.app/

## Dataset

Global Crunchbase-sourced startup dataset (~66,000 companies, 100+ countries,
1902-2016), filtered to companies with a resolved outcome (closed, acquired, or IPO).
"Still operating" companies are excluded since that outcome isn't yet resolved.

Source: [Big Startup Success/Fail Dataset from Crunchbase](https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase) (Kaggle)

## Model

- **Algorithm:** Logistic regression (scikit-learn), tuned via Optuna (regularization
  strength, L1/L2 blend, class weighting) with 5-fold stratified cross-validation
- **Final performance:** 75% accuracy, 0.82 ROC-AUC, balanced precision/recall
  (~0.73-0.76) across both classes, evaluated on a 2,667-company held-out test set

## Key Features

- Funding: total raised (log-scaled), number of rounds, funding velocity
  (rounds per year since founding), time between first and last round
- Timing: years to most recent funding round
- Location: continent (mapped from country code)
- Industry: primary category (top 10 categories + "Other")
- Missing-data flags: funding and founding-date completeness, since incomplete
  records correlate meaningfully with company outcome

## Tech Stack

- Python, pandas, scikit-learn, Optuna
- Streamlit (app + hosting)
- pycountry / pycountry-convert (geography mapping)

## Project Structure
startup-predictor/
├── app/
│ └── app.py # Streamlit app
├── src/
│ ├── preprocess.py # Data cleaning and feature engineering
│ ├── train.py # Baseline model training
│ └── tune.py # Optuna hyperparameter tuning
├── data/
│ └── raw/ # Dataset (not tracked in git)
├── models/
│ └── tuned_logreg.pkl # Final trained model
├── .streamlit/
│ └── config.toml # App theme config
└── requirements.txt

## Known Limitations

- Restricting to resolved outcomes skews the training data toward older companies
- North America still makes up roughly two-thirds of the data even globally;
  predictions for underrepresented regions (e.g. Africa, ~20 companies) are
  low-confidence
- Tested whether adding nonlinear and interaction terms (squared timing
  features, funding x region interactions) would improve performance, and
  found only a negligible gain, (+0.001 AUC) indicating the model's
  performance ceiling comes from feature richness