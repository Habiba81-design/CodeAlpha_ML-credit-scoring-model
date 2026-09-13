# CodeAlpha_CreditScoringModel

Machine Learning internship project: CodeAlpha (Task 1: Credit Scoring Model).

## Problem

Predict whether a borrower will experience serious financial delinquency within
the next two years, using their credit history and financial profile. This is a
binary classification problem trained on the ["Give Me Some Credit"](https://www.kaggle.com/c/GiveMeSomeCredit)
dataset (Kaggle, 2011) — 150,000 real, anonymized consumer credit records.

## Project structure

```
CodeAlpha_CreditScoringModel/
├── 01_eda.py              # Exploratory data analysis (run first)
├── 02_train_model.py      # Cleaning, feature engineering, training, evaluation
├── 03_predict.py          # Score new applicants with the saved model
├── requirements.txt
├── README.md
├── cs-training.csv        # (not committed — see Data section)
├── cs-test.csv            # (not committed — see Data section)
└── outputs/
    ├── eda/                # plots from 01_eda.py
    └── model/              # trained model, metrics, plots, test predictions
```

## Data

Download `cs-training.csv` and `cs-test.csv` from the
[Kaggle competition page](https://www.kaggle.com/c/GiveMeSomeCredit/data) and
place them in the project root. They are not committed to this repo due to
size/licensing — this is standard practice for Kaggle datasets.

## Approach

1. **EDA** — identified severe class imbalance (6.7% default rate), missing
   values in `MonthlyIncome` (~20%) and `NumberOfDependents` (~2.6%), and
   real-world data quality issues: an impossible `age = 0`, sentinel/error
   codes (96, 98) in the "days past due" columns, and extreme outliers in
   utilization ratio and debt ratio.
2. **Cleaning** — sentinel and impossible values converted to missing (not
   deleted), then imputed; outliers capped at the 99th percentile rather than
   dropped, to preserve every applicant's record.
3. **Feature engineering** — missingness flags, total past-due incidents,
   income-per-dependent, and credit-lines-per-age.
4. **Modeling** — Logistic Regression (interpretable baseline), Random
   Forest, and HistGradientBoosting (sklearn's native, dependency-free
   gradient boosting), each tuned via `RandomizedSearchCV` scored on
   ROC-AUC, with `class_weight="balanced"` to handle the imbalance.
5. **Evaluation** — ROC-AUC and PR-AUC as primary metrics (not accuracy,
   which is misleading on this imbalanced data), plus precision/recall/F1
   and confusion matrices on a held-out validation split.

## Results

| Model | CV ROC-AUC | Validation ROC-AUC | Validation PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8543 | 0.8547 | 0.3817 | 0.2135 | 0.7572 | 0.3331 |
| Random Forest | 0.8628 | 0.8629 | 0.3968 | 0.2354 | 0.7423 | 0.3574 |
| **HistGradientBoosting (best)** | **0.8637** | **0.8648** | **0.4034** | 0.2131 | **0.7862** | 0.3353 |

Best model: **HistGradientBoosting**, selected by highest validation ROC-AUC (0.8648).

**Interpretation:** ROC-AUC of ~0.86 is competitive with the top entries in the
original 2011 Kaggle competition. Recall was prioritized in this project's
threshold choice (0.5 default) — the model catches ~79% of actual defaulters,
which matters more than raw precision in a lending context, where missing a
defaulter is typically costlier than a false alarm on a safe borrower. This
came at the cost of precision (~21%), a deliberate, explainable tradeoff
rather than a weakness of the model.

## How to run

```bash
pip install -r requirements.txt
python3 01_eda.py            # explore the data, saves plots to outputs/eda/
python3 02_train_model.py    # clean, train, tune, evaluate, save the model
python3 03_predict.py new_applicants.csv predictions.csv   # score new data
streamlit run app.py         # interactive demo — enter one applicant, get a live prediction
```

## Notes

- `xgboost` was intentionally not used to keep the project dependency-light;
  `HistGradientBoostingClassifier` is a strong, native scikit-learn
  alternative that handles missing values internally.
- All preprocessing (imputation, scaling) is fit only on the training fold
  inside a scikit-learn `Pipeline`, to avoid data leakage into validation
  or test predictions.
