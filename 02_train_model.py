"""
02_train_model.py
Full training pipeline for the "Give Me Some Credit" credit scoring model.

Steps:
  1. Load data
  2. Clean (fix bad values, sentinel codes, duplicates)
  3. Feature engineering
  4. Train/validation split (stratified)
  5. Preprocessing pipeline (impute + scale), fit on train fold only
  6. Train 3 models: Logistic Regression, Random Forest, HistGradientBoosting
  7. Hyperparameter tuning via RandomizedSearchCV (scored on ROC-AUC)
  8. Evaluate on held-out validation set (ROC-AUC, PR-AUC, precision/recall/F1)
  9. Pick the best model, retrain on full training data
  10. Score the Kaggle test set (unlabeled) and save predictions
  11. Save the fitted pipeline + model to disk

Run 01_eda.py first to understand the data before running this.
"""

import os
import json
import joblib
import numpy as np
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_PATH = "cs-training.csv"
TEST_PATH = "cs-test.csv"
OUT_DIR = "outputs/model"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "SeriousDlqin2yrs"
RANDOM_STATE = 42
N_SEARCH_ITER = 15  # RandomizedSearchCV iterations per model; raise for a more thorough search

# Use all cores if more than one is available, otherwise stay single-threaded.
# (Parallel joblib workers can be unstable / get killed on single-core, low-memory boxes.)
N_JOBS = -1 if multiprocessing.cpu_count() > 1 else 1


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_data():
    train_df = pd.read_csv(TRAIN_PATH, index_col=0)
    test_df = pd.read_csv(TEST_PATH, index_col=0)
    return train_df, test_df


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()

    # Drop exact duplicate rows (train only -- never drop rows from the
    # Kaggle test set, every row there needs a prediction).
    if is_train:
        before = len(df)
        df = df.drop_duplicates()
        print(f"Dropped {before - len(df)} duplicate rows")

    # age == 0 is impossible; treat as missing and impute with median later.
    df.loc[df["age"] == 0, "age"] = np.nan

    # The three "days late" columns use 96/98 as sentinel/placeholder codes,
    # not real counts. Treat as missing rather than literal huge values.
    sentinel_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    for c in sentinel_cols:
        df.loc[df[c].isin([96, 98]), c] = np.nan

    # Cap extreme outliers (winsorize at 99th percentile) rather than deleting
    # rows -- keeps the signal that "this applicant is very high risk" while
    # preventing a handful of rows from dominating the loss function.
    for c in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]:
        cap = df[c].quantile(0.99)
        df[c] = df[c].clip(upper=cap)

    return df


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Missingness flags -- whether income/dependents were disclosed can
    # itself carry signal, so preserve it before imputing.
    df["MonthlyIncome_missing"] = df["MonthlyIncome"].isna().astype(int)
    df["NumberOfDependents_missing"] = df["NumberOfDependents"].isna().astype(int)

    # Total past-due incidents across all three severity buckets.
    df["TotalPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"].fillna(0)
        + df["NumberOfTime60-89DaysPastDueNotWorse"].fillna(0)
        + df["NumberOfTimes90DaysLate"].fillna(0)
    )

    # Income per household member (dependents + self). +1 avoids division by zero.
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"].fillna(0) + 1)

    # Credit lines relative to age -- a rough proxy for credit history depth
    # relative to how long the person has been able to build one.
    df["CreditLinesPerAge"] = df["NumberOfOpenCreditLinesAndLoans"] / df["age"].replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# 4-5. Split + preprocessing pipeline
# ---------------------------------------------------------------------------
def build_preprocessor(feature_cols):
    """
    Median-impute all numeric features, then scale.
    Scaling is unnecessary for tree models but required for Logistic
    Regression -- applying it to all models keeps one shared pipeline and
    does no harm to the tree-based models.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, feature_cols),
    ])
    return preprocessor


# ---------------------------------------------------------------------------
# 6-7. Models + hyperparameter search spaces
# ---------------------------------------------------------------------------
def get_model_configs():
    """
    Each entry: (name, estimator, param_distributions for RandomizedSearchCV)
    class_weight="balanced" / equivalent handles the ~14:1 class imbalance
    without needing SMOTE or imblearn.
    """
    configs = {
        "logistic_regression": {
            "estimator": LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
            ),
            "param_distributions": {
                "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS
            ),
            "param_distributions": {
                "model__n_estimators": [200, 300, 400],
                "model__max_depth": [6, 10, 14, None],
                "model__min_samples_leaf": [1, 5, 10, 20],
                "model__max_features": ["sqrt", "log2"],
            },
        },
        "hist_gradient_boosting": {
            # sklearn's native gradient boosting -- handles missing values
            # natively and is a strong, dependency-free stand-in for XGBoost.
            # If xgboost is available in your environment, swap this for
            # xgboost.XGBClassifier(scale_pos_weight=..., eval_metric="auc").
            "estimator": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_distributions": {
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [4, 6, 8, None],
                "model__max_iter": [150, 250, 350],
                "model__l2_regularization": [0.0, 0.1, 1.0],
                "model__class_weight": ["balanced"],
            },
        },
    }
    return configs


def train_and_tune(name, estimator, param_distributions, preprocessor, X_train, y_train, cv):
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", estimator),
    ])

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=N_SEARCH_ITER,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=0,
    )
    print(f"\nTuning {name} ...")
    search.fit(X_train, y_train)
    print(f"  Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_estimator_, search.best_score_


# ---------------------------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(name, fitted_pipe, X_val, y_val, threshold=0.5):
    proba = fitted_pipe.predict_proba(X_val)[:, 1]
    preds = (proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_val, proba)
    pr_auc = average_precision_score(y_val, proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_val, preds)

    print(f"\n--- {name} (validation, threshold={threshold}) ---")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{cm}")

    return {
        "name": name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "proba": proba,
    }


def plot_roc_pr_curves(results, y_val):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for res in results:
        fpr, tpr, _ = roc_curve(y_val, res["proba"])
        axes[0].plot(fpr, tpr, label=f"{res['name']} (AUC={res['roc_auc']:.3f})")

        prec, rec, _ = precision_recall_curve(y_val, res["proba"])
        axes[1].plot(rec, prec, label=f"{res['name']} (AUC={res['pr_auc']:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves")
    axes[0].legend()

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "roc_pr_curves.png"), dpi=120)
    plt.close(fig)


def plot_feature_importance(fitted_pipe, feature_cols, model_name):
    model = fitted_pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_cols[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#4C72B0",
    )
    ax.set_title(f"Feature Importance — {model_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"feature_importance_{model_name}.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    train_df, test_df = load_data()

    # --- Clean + engineer ---
    train_df = clean_data(train_df, is_train=True)
    test_df = clean_data(test_df, is_train=False)

    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    feature_cols = [c for c in train_df.columns if c != TARGET]

    X = train_df[feature_cols]
    y = train_df[TARGET]

    # --- Split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")

    preprocessor = build_preprocessor(feature_cols)
    # 3-fold CV for the hyperparameter search keeps runtime reasonable on a
    # normal laptop given 150k rows x 3 models x N_SEARCH_ITER candidates.
    # Bump to 5 for a more robust (but slower) search.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # --- Train + tune each model ---
    configs = get_model_configs()
    fitted_models = {}
    cv_scores = {}
    for name, cfg in configs.items():
        best_pipe, best_cv_score = train_and_tune(
            name, cfg["estimator"], cfg["param_distributions"],
            preprocessor, X_train, y_train, cv,
        )
        fitted_models[name] = best_pipe
        cv_scores[name] = best_cv_score

    # --- Evaluate all models on the held-out validation set ---
    results = []
    for name, pipe in fitted_models.items():
        results.append(evaluate_model(name, pipe, X_val, y_val))

    plot_roc_pr_curves(results, y_val)

    # --- Pick best model by validation ROC-AUC ---
    best_result = max(results, key=lambda r: r["roc_auc"])
    best_name = best_result["name"]
    best_pipe = fitted_models[best_name]
    print(f"\n=== Best model: {best_name} (val ROC-AUC={best_result['roc_auc']:.4f}) ===")

    plot_feature_importance(best_pipe, feature_cols, best_name)

    # --- Retrain best model on the FULL training data (train+val combined) ---
    print(f"\nRetraining {best_name} on full training data ...")
    best_pipe.fit(X, y)

    # --- Score the Kaggle test set (unlabeled) ---
    X_test = test_df[feature_cols]
    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame({
        "Id": test_df.index,
        "Probability": test_proba,
    })
    submission_path = os.path.join(OUT_DIR, "test_predictions.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Test predictions saved to: {submission_path}")

    # --- Save model + metadata ---
    model_path = os.path.join(OUT_DIR, "credit_scoring_pipeline.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"Model saved to: {model_path}")

    metadata = {
        "best_model": best_name,
        "feature_cols": feature_cols,
        "cv_roc_auc": cv_scores,
        "validation_metrics": {
            r["name"]: {
                "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
                "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
            } for r in results
        },
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {os.path.join(OUT_DIR, 'metadata.json')}")


if __name__ == "__main__":
    main()
