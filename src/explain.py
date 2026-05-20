from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor, Pool
from sklearn.inspection import PartialDependenceDisplay

from src.data import DEFAULT_SCHEMA, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/cleaned_data_house_prices.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--figures", default="reports/figures")
    parser.add_argument("--max-rows", type=int, default=2000)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)
    figures_dir = Path(args.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((artifacts_dir / "model_meta.json").read_text(encoding="utf-8"))

    model = CatBoostRegressor()
    model.load_model(artifacts_dir / "catboost_house_price.cbm")

    df = load_dataset(args.dataset, schema=DEFAULT_SCHEMA)

    feature_cols = meta["feature_columns"]
    cat_cols = meta["categorical_columns"]
    target = meta["target"]

    sample = df.sample(n=min(args.max_rows, len(df)), random_state=42).reset_index(drop=True)

    x = sample[feature_cols].copy()
    for c in cat_cols:
        x[c] = x[c].astype(str).fillna("__MISSING__")

    cat_idx = [feature_cols.index(c) for c in cat_cols]
    pool = Pool(x, cat_features=cat_idx)

    # Global feature importance (CatBoost)
    fi = model.get_feature_importance(type="FeatureImportance")
    order = np.argsort(fi)[::-1]

    plt.figure(figsize=(9, 6))
    top_k = min(15, len(feature_cols))
    idx = order[:top_k][::-1]
    plt.barh([feature_cols[i] for i in idx], [fi[i] for i in idx])
    plt.title("Top Feature Importances (CatBoost)")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=220)
    plt.close()

    # SHAP summary plot (TreeExplainer)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, x, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary.png", dpi=220, bbox_inches="tight")
    plt.close()

    # Partial dependence plots for a couple of strong numeric features (if present).
    numeric_candidates = [c for c in meta.get("numeric_columns", []) if c in feature_cols]
    pdp_features = []
    for c in ["House size", "Land size", "Beds", "Baths"]:
        if c in numeric_candidates:
            pdp_features.append(c)
    if len(pdp_features) < 2 and numeric_candidates:
        pdp_features = numeric_candidates[:2]

    if pdp_features:
        PartialDependenceDisplay.from_estimator(model, x, pdp_features, kind="average")
        plt.tight_layout()
        plt.savefig(figures_dir / "pdp.png", dpi=220, bbox_inches="tight")
        plt.close()

    print("Saved explainability figures to:", figures_dir)


if __name__ == "__main__":
    main()
