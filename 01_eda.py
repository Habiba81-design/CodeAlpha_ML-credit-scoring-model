"""
01_eda.py
Exploratory Data Analysis for the "Give Me Some Credit" credit scoring dataset.

Run this first to understand the data before any cleaning or modeling.
Outputs:
  - Console summary (class balance, missingness, outlier counts, duplicates)
  - PNG plots saved to ./outputs/eda/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_PATH = "cs-training.csv"
OUT_DIR = "outputs/eda"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "SeriousDlqin2yrs"


def load_data(path: str) -> pd.DataFrame:
    """Load the training data, using the first column as the row index."""
    df = pd.read_csv(path, index_col=0)
    return df


def print_overview(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("SHAPE:", df.shape)
    print("=" * 70)

    print("\n--- DTYPES ---")
    print(df.dtypes)

    print("\n--- DUPLICATE ROWS ---")
    print(f"{df.duplicated().sum()} exact duplicate rows found")

    print("\n--- CLASS BALANCE (target) ---")
    print(df[TARGET].value_counts(normalize=True).rename("proportion"))

    print("\n--- MISSING VALUES ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print(pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct}))

    print("\n--- DESCRIBE (numeric) ---")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.describe().T)


def flag_known_data_issues(df: pd.DataFrame) -> None:
    """
    Print counts for the specific data-quality issues known to exist in this
    dataset. These are NOT fixed here -- this script is read-only / diagnostic.
    Cleaning happens in 02_train_model.py.
    """
    print("\n--- KNOWN DATA QUALITY ISSUES ---")

    print("age == 0 (impossible):", (df["age"] == 0).sum())

    util_col = "RevolvingUtilizationOfUnsecuredLines"
    print(f"{util_col} > 1 (should be a ratio <= 1):", (df[util_col] > 1).sum())
    print(f"{util_col} > 10 (extreme outliers):", (df[util_col] > 10).sum())

    sentinel_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    for c in sentinel_cols:
        n_sentinel = df[c].isin([96, 98]).sum()
        print(f"{c}: {n_sentinel} rows with sentinel values (96/98)")

    print("DebtRatio max value:", df["DebtRatio"].max(), "(extreme outliers likely)")


def plot_distributions(df: pd.DataFrame) -> None:
    """Histogram of every numeric feature, saved as a single grid image."""
    numeric_cols = df.columns.tolist()
    n = len(numeric_cols)
    ncols = 3
    nrows = -(-n // ncols)  # ceil division

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        # Clip to 99th percentile purely for readable plotting (not for modeling)
        data = df[col].dropna()
        upper = data.quantile(0.99)
        axes[i].hist(data.clip(upper=upper), bins=50, color="#4C72B0", edgecolor="none")
        axes[i].set_title(col, fontsize=10)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "feature_distributions.png"), dpi=120)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame) -> None:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "correlation_matrix.png"), dpi=120)
    plt.close(fig)


def plot_target_relationship(df: pd.DataFrame) -> None:
    """Boxplots of a few key features split by target class."""
    features = ["RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio", "MonthlyIncome"]
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 5))

    for i, col in enumerate(features):
        data = df[[col, TARGET]].dropna()
        upper = data[col].quantile(0.99)
        data = data[data[col] <= upper]
        groups = [data.loc[data[TARGET] == 0, col], data.loc[data[TARGET] == 1, col]]
        axes[i].boxplot(groups, tick_labels=["No Default", "Default"])
        axes[i].set_title(col)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "target_relationship_boxplots.png"), dpi=120)
    plt.close(fig)


def main():
    df = load_data(TRAIN_PATH)
    print_overview(df)
    flag_known_data_issues(df)
    plot_distributions(df)
    plot_correlation(df)
    plot_target_relationship(df)
    print(f"\nPlots saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
