# Startup Outcome Probability Model

Estimates whether an early-stage startup is headed toward acquisition or IPO, versus closure, using only funding history, timing, industry, and location, plus a bonus valuation estimator for high-confidence, well-funded cases.

**[Try the live app](https://startup-success-predictor-gxewqlkcddbv7xme9qwedb.streamlit.app/)** · **[View GitHub Page](https://savirpatil.github.io/startup-predictor/)**

| | |
|---|---|
| ~66,000 | Companies |
| 100+ | Countries |
| 75% | Accuracy |
| 0.823 | ROC-AUC |

## Overview

Most startups fail, and the reasons can be difficult to quantify. This project explores whether a simple, interpretable model can predict a startup's likely outcome using only information available in the early stages, while showing which factors are driving that prediction.

## Repository Structure
startup-predictor/
├── data/
│ └── raw/ Raw dataset (gitignored)
├── src/
│ ├── preprocess.py Data cleaning & feature engineering
│ ├── train.py Baseline model training
│ └── tune.py Optuna hyperparameter tuning
├── models/
│ └── tuned_logreg.pkl Final trained model
├── app/
│ └── app.py Streamlit demo app
├── .streamlit/
│ └── config.toml App theme config
├── requirements.txt
└── README.md
## Setup

```bash
git clone <your-repo-url>
cd startup-predictor
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the dataset at `data/raw/big_startup_secsees_dataset.csv`.

## Usage

Train the baseline model:
```bash
python src/train.py
```

Tune and produce the final deployed model:
```bash
python src/tune.py
```

Launch the demo app:
```bash
streamlit run app/app.py
```

## Model

Logistic regression is a supervised, binary-classification algorithm. Given a set of inputs, it outputs the probability a company falls into one of two classes: **successful** (acquired or IPO'd) or **closed**.

Tested against random forest and neural network models under the same cross-validation setup, logistic regression matched or beat both, with the added benefit of fully interpretable coefficients.

**Inputs**
- *Funding:* total funding raised, number of funding rounds
- *Timing:* years to first round, years to most recent round
- *Profile:* headquarters region, primary industry category

**Output:** A success probability, plus the top factors pushing that estimate toward successful or toward closed.

### Evolution

| Stage | Description | AUC |
|---|---|---|
| Baselines | Discarded a synthetic-data version, then a small 923-company US-only dataset | 0.80 |
| Global rebuild | Rebuilt on a 13,000+ company global dataset with resolved outcomes | 0.82 |
| Tuned, final | Optuna-tuned, multicollinearity fixed — this is the deployed model | 0.82 |
| Bonus model | An XGBoost valuation estimator for high-confidence, well-funded cases | R² 0.51 |

## Evaluation

| Metric | Value |
|---|---|
| Accuracy | 75% |
| ROC-AUC | 0.823 |
| Precision / recall | 0.73–0.76 |

Tuned with Optuna across regularization strength, L1/L2 blend, and class weighting, under 5-fold stratified cross-validation.

**Two tuning runs, same ceiling.** Independent runs on slightly different feature sets converged on nearly identical AUC (0.8225 vs. 0.8226) despite landing on different hyperparameters, suggesting the model had reached the ceiling of this feature set.

Performance held steady or improved across every iteration, even as the classification task got objectively harder (less class imbalance to lean on).

## Bonus: Valuation Estimator

As a bonus feature, when the primary model is highly confident a company is on track for success, a second model estimates a possible valuation range using the same company profile. This model is trained separately on a dataset of unicorns — startups that have already reached a billion-dollar valuation — so it is only used for the predictions the primary model is most confident in.

**When it activates:** requires at least 75% success probability from the primary model, and at least $10M in total funding. Between $10M and $100M, a low-confidence warning is shown, since 96% of the training data raised $100M or more. Above $100M, the estimate is shown without a warning.

| Metric | Value |
|---|---|
| R² score | 0.51 |
| Mean absolute error | $1.51B |
| Median absolute error | $652M |

**Strongest drivers:** funding relative to industry average and total funding raised together account for over half of the model's predictive weight; region and sector signals contribute smaller amounts.

## Missing Data as Signal

Closed startups are missing key fields (founding date, country, funding amount) at roughly 2x the rate of successful ones. Rather than dropping incomplete records, we kept them and flagged what's missing. The pattern shows up directly in the final model: missing-data flags carry a negative association with success.

**Counterintuitive finding:** startups raising funding rounds more frequently were less likely to succeed in our model — possibly reflecting urgent cash needs rather than momentum.

## Data Sources

Filtered to companies with a resolved outcome (closed, acquired, or IPO'd). Still-operating companies are excluded, since that outcome hasn't happened yet.

- Primary: [Crunchbase dataset](https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase)
- Secondary: unicorn dataset (for the valuation estimator)

## Bias & Impact

**Essential question:** How can the process of creating AI/ML solutions amplify or mitigate bias?

**Positive**
- Supports founders and investors with data-driven insight, grounded in real patterns
- Explains which factors influenced each prediction, instead of acting as a black box

**Negative**
- Historical funding data can reinforce existing investment biases if used uncritically

The original model used almost entirely US data. The global rebuild reduced that imbalance but didn't remove it: North America still makes up roughly two-thirds of the data, so predictions for underrepresented regions, like Africa (about 20 companies), should be treated as low-confidence.

## Limitations

- **Recency bias.** Resolved-outcomes-only skews training data toward older startups.
- **Regional imbalance.** North America still makes up roughly two-thirds of the data.
- **Feature ceiling.** Nonlinear and interaction terms gained only +0.001 AUC; more model complexity isn't the next lever.
- **Unobserved factors.** Team dynamics, timing, and luck aren't captured by funding or location data.

## Next Steps

- Expand the dataset with more recent, globally balanced startup records *(ongoing)*
- Add founder, market, and economic features currently missing from the model *(scope pending)*
- Test calibration and fairness across regions before any broader deployment *(expected Sept 2026)*

## Built With

Python · pandas · scikit-learn · Optuna · XGBoost · Streamlit · pycountry

## Citations

- Argaw, Y. M., & Liu, Y. (2024). The pathway to startup success: A comprehensive systematic review of critical factors and the future research agenda in developed and emerging markets. *Systems, 12*(12), 541.
- Founders Forum Group. (2025, May 13). *The ultimate startup guide with statistics (2024-2025)*. Founders Forum.
- Okrah, J., Nepp, A., & Agbozo, E. (2018). Exploring the factors of startup success and growth. *The Business & Management Review, 9*(3), 229-237.
- Potanin, M., Chertok, A., Zorin, K., & Shtabtsovsky, C. (2023). Startup success prediction and VC portfolio simulation using CrunchBase data. *arXiv*.
- Razaghzadeh Bidgoli, M., Raeesi Vanani, I., & Goodarzi, M. (2024). Predicting the success of startups using a machine learning approach. *Journal of Innovation and Entrepreneurship, 13*(1), 80.
- Wang, I. (2019). *Predicting a startup's acquisition status* (CS229 machine learning technical report). Stanford University.
- Żbikowski, K., & Antosiuk, P. (2021). A machine learning, bias-free approach for predicting business success using Crunchbase data. *Information Processing & Management, 58*(4), 102555.
