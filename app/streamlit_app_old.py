from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import joblib

try:
    from catboost import CatBoostRegressor, Pool
except Exception:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore
    Pool = None  # type: ignore


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def load_model_and_meta(artifacts_dir: Path):
    meta = json.loads((artifacts_dir / "model_meta.json").read_text(encoding="utf-8"))

    # Prefer sklearn pipeline if present (more reliable in this workspace).
    sklearn_path = artifacts_dir / "hgb_house_price.joblib"
    if sklearn_path.exists():
        model = joblib.load(sklearn_path)
        return model, meta

    # Fallback to CatBoost if available.
    if CatBoostRegressor is None:
        raise RuntimeError("CatBoost is not available and no sklearn model artifact was found.")
    model = CatBoostRegressor()
    model.load_model(artifacts_dir / "catboost_house_price.cbm")
    return model, meta


def make_input_form(meta: dict) -> dict:
    inputs: dict = {}

    with st.sidebar:
        st.header("Property Details")

        for col in meta["feature_columns"]:
            if col in meta.get("categorical_columns", []):
                options = meta.get("categories", {}).get(col, [])
                if options:
                    inputs[col] = st.selectbox(col, options=options, index=0)
                else:
                    inputs[col] = st.text_input(col, value="")
            else:
                # Numeric
                default = 0.0
                if col in {"Beds", "Baths"}:
                    inputs[col] = st.number_input(col, min_value=0, max_value=20, value=3, step=1)
                else:
                    inputs[col] = st.number_input(col, value=float(default), step=1.0)

        st.caption("Tip: If you don't know a value, leave it as 0 or pick the most common category.")

    return inputs


def inputs_to_dataframe(inputs: dict, feature_columns: list[str], categorical_columns: list[str]) -> pd.DataFrame:
    row = {}
    for col in feature_columns:
        if col in categorical_columns:
            row[col] = str(inputs.get(col, "__MISSING__") or "__MISSING__").strip()
        else:
            val = inputs.get(col, 0.0)
            try:
                row[col] = float(val)
            except Exception:
                row[col] = np.nan

    return pd.DataFrame([row], columns=feature_columns)


def explain_single(model: CatBoostRegressor, x_row: pd.DataFrame, *, cat_cols: list[str], feature_cols: list[str]) -> tuple[list[str], np.ndarray]:
    cat_idx = [feature_cols.index(c) for c in cat_cols]
    pool = Pool(x_row, cat_features=cat_idx)
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    # CatBoost returns shap values with last column = expected value
    shap_contrib = np.array(shap_values[0][:-1], dtype=float)
    return feature_cols, shap_contrib


def main() -> None:
    st.set_page_config(page_title="Sri Lanka House Price Predictor", layout="wide")

    st.title("Sri Lanka House Price Predictor")
    st.write(
        "Predict house prices using a gradient-boosted decision tree model (CatBoost) trained on Sri Lankan property data."
    )

    model, meta = load_model_and_meta(ARTIFACTS_DIR)

    inputs = make_input_form(meta)

    feature_cols = meta["feature_columns"]
    cat_cols = meta.get("categorical_columns", [])

    x_row = inputs_to_dataframe(inputs, feature_cols, cat_cols)

    model_type = meta.get("model_type", "catboost")

    if model_type == "sklearn_hgb_onehot":
        pred = float(model.predict(x_row)[0])
    else:
        cat_idx = [feature_cols.index(c) for c in cat_cols]
        pool = Pool(x_row, cat_features=cat_idx)
        raw_pred = model.predict(pool)[0]
        transform = meta.get("target_transform", meta.get("notes", {}).get("target_transform", "none"))
        pred = float(np.expm1(raw_pred) if transform == "log1p" else raw_pred)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Prediction")
        st.metric("Estimated Price (LKR)", f"{pred:,.0f}")
        st.caption("This is a point estimate; real prices vary by condition, timing, and negotiation.")

    with c2:
        st.subheader("Model Quality")
        tm = meta.get("test_metrics", {})
        st.write(
            {
                "R² (test)": round(float(tm.get("r2", 0.0)), 4),
                "RMSE (test)": round(float(tm.get("rmse", 0.0)), 2),
                "MAE (test)": round(float(tm.get("mae", 0.0)), 2),
                "MAPE% (test)": round(float(tm.get("mape_percent", 0.0)), 2),
            }
        )

    st.divider()

    tab_pred, tab_explain, tab_importance = st.tabs(["Explain This Prediction", "Feature Contributions", "Global Importance"])

    with tab_pred:
        st.write(
            "Use the tabs to understand what the model is using to make this prediction. "
            "Explanations are based on SHAP values from the trained CatBoost model."
        )

    with tab_explain:
        st.subheader("Top Feature Contributions (SHAP)")
        if model_type != "sklearn_hgb_onehot" and st.button("Generate Explanation"):
            names, contrib = explain_single(model, x_row, cat_cols=cat_cols, feature_cols=feature_cols)
            order = np.argsort(np.abs(contrib))[::-1][:10]

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh([names[i] for i in order][::-1], [contrib[i] for i in order][::-1])
            ax.axvline(0.0, color="black", linewidth=1)
            ax.set_xlabel("Contribution to prediction (model units)")
            ax.set_title("Top 10 SHAP Contributions (local)")
            st.pyplot(fig, clear_figure=True)
            st.caption(
                "Positive values push the prediction up; negative values push it down. "
                "If the target was trained in log-space, units are log(LKR)."
            )
        if model_type == "sklearn_hgb_onehot":
            st.info("Local SHAP is enabled for CatBoost runs; this sklearn model uses global explanations (PDP + importance) in the report.")

    with tab_importance:
        st.subheader("Global Feature Importance")
        if model_type == "sklearn_hgb_onehot":
            st.write("See `reports/figures/feature_importance.png` after running explainability generation.")
        else:
            fi = model.get_feature_importance(type="FeatureImportance")
            order = np.argsort(fi)[::-1][:15]

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh([feature_cols[i] for i in order][::-1], [fi[i] for i in order][::-1])
            ax.set_title("Top 15 Feature Importances")
            st.pyplot(fig, clear_figure=True)


if __name__ == "__main__":
    main()
