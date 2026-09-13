"""
03_predict.py
Score new applicants using the model trained by 02_train_model.py.

Usage:
    python3 03_predict.py new_applicants.csv predictions_out.csv

The input CSV must have the same raw columns as the original training data
(everything except the target column, SeriousDlqin2yrs). This script applies
the exact same cleaning + feature engineering + preprocessing used at
training time, so predictions stay consistent with how the model was built.
"""

import sys
import importlib.util
import joblib
import pandas as pd

MODEL_PATH = "outputs/model/credit_scoring_pipeline.joblib"


def load_pipeline_module():
    """Import 02_train_model.py as a module to reuse its clean/engineer functions."""
    spec = importlib.util.spec_from_file_location("train_module", "02_train_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 03_predict.py <input_csv> <output_csv>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    train_module = load_pipeline_module()
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(input_path, index_col=0)
    df = train_module.clean_data(df, is_train=False)
    df = train_module.engineer_features(df)

    feature_cols = [c for c in df.columns if c != "SeriousDlqin2yrs"]
    proba = model.predict_proba(df[feature_cols])[:, 1]

    out = pd.DataFrame({"Id": df.index, "DefaultProbability": proba})
    out.to_csv(output_path, index=False)
    print(f"Saved {len(out)} predictions to {output_path}")


if __name__ == "__main__":
    main()
