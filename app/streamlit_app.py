from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

try:
    from catboost import CatBoostRegressor, Pool
except Exception:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore[assignment]
    Pool = None  # type: ignore[assignment]


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATASET_PATH = ROOT_DIR / "dataset" / "cleaned_data_house_prices.csv"


@dataclass(frozen=True)
class PredictionResult:
    value: float
    lower: float
    upper: float
    model_type: str


def format_lkr(value: float, *, decimals: int = 1) -> str:
    if not np.isfinite(value):
        return "LKR 0"
    value = max(float(value), 0.0)
    if value >= 1_000_000:
        return f"LKR {value / 1_000_000:,.{decimals}f}M"
    return f"LKR {value:,.0f}"


def format_metric(value: float, suffix: str = "") -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:,.1f}{suffix}"


@st.cache_resource(show_spinner=False)
def load_model_and_meta(artifacts_dir: Path) -> tuple[Any, dict[str, Any]]:
    meta_path = artifacts_dir / "model_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    sklearn_path = artifacts_dir / "hgb_house_price.joblib"
    if sklearn_path.exists():
        return joblib.load(sklearn_path), meta

    model_path = artifacts_dir / "catboost_house_price.cbm"
    if CatBoostRegressor is None or not model_path.exists():
        raise RuntimeError("No usable model artifact was found in the artifacts folder.")

    model = CatBoostRegressor()
    model.load_model(model_path)
    return model, meta


@st.cache_data(show_spinner=False)
def load_market_data(dataset_path: Path) -> pd.DataFrame:
    from src.data import DEFAULT_SCHEMA, load_dataset

    df = load_dataset(dataset_path, schema=DEFAULT_SCHEMA)
    keep_cols = [
        "Price",
        "Baths",
        "Land size",
        "Beds",
        "House size",
        "Seller_type",
        "town",
        "district",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    for col in ("district", "town", "Seller_type"):
        if col in df:
            df[col] = df[col].astype(str).str.strip()

    for col in ("Price", "Baths", "Land size", "Beds", "House size"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["Price"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def district_town_map(dataset_path: Path, categories: dict[str, list[str]]) -> dict[str, list[str]]:
    try:
        df = load_market_data(dataset_path)
        mapping: dict[str, list[str]] = {}
        for district, group in df.groupby("district", sort=True):
            towns = group["town"].dropna().astype(str).str.strip()
            mapping[str(district)] = towns.value_counts().index.tolist()
        return mapping
    except Exception:
        towns = categories.get("town", [])
        return {district: towns for district in categories.get("district", [])}


def inputs_to_dataframe(
    inputs: dict[str, Any],
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for col in feature_columns:
        if col in categorical_columns:
            row[col] = str(inputs.get(col, "__MISSING__") or "__MISSING__").strip()
            continue

        try:
            row[col] = float(inputs.get(col, 0.0))
        except Exception:
            row[col] = np.nan

    return pd.DataFrame([row], columns=feature_columns)


def predict_price(model: Any, meta: dict[str, Any], x_row: pd.DataFrame) -> PredictionResult:
    feature_cols = meta["feature_columns"]
    cat_cols = meta.get("categorical_columns", [])
    model_type = meta.get("model_type", "catboost")

    if model_type == "sklearn_hgb_onehot":
        pred = float(model.predict(x_row)[0])
    else:
        if Pool is None:
            raise RuntimeError("CatBoost is required to make predictions with this artifact.")
        cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
        pool = Pool(x_row, cat_features=cat_idx)
        raw_pred = float(model.predict(pool)[0])
        transform = meta.get("target_transform", meta.get("notes", {}).get("target_transform", "none"))
        pred = float(np.expm1(raw_pred) if transform == "log1p" else raw_pred)

    mae = float(meta.get("test_metrics", {}).get("mae", 0.0) or 0.0)
    spread = max(mae, pred * 0.12)
    return PredictionResult(
        value=max(pred, 0.0),
        lower=max(pred - spread, 0.0),
        upper=max(pred + spread, 0.0),
        model_type=model_type,
    )


def local_contributions(
    model: Any,
    x_row: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    if Pool is None:
        return pd.DataFrame(columns=["feature", "contribution"])

    cat_idx = [feature_columns.index(c) for c in categorical_columns if c in feature_columns]
    pool = Pool(x_row, cat_features=cat_idx)
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    contrib = np.array(shap_values[0][:-1], dtype=float)

    out = pd.DataFrame({"feature": feature_columns, "contribution": contrib})
    out["abs_contribution"] = out["contribution"].abs()
    return out.sort_values("abs_contribution", ascending=False)


def market_summary(df: pd.DataFrame, district: str, town: str) -> tuple[pd.DataFrame, str]:
    town_df = df[(df["district"] == district) & (df["town"] == town)].copy()
    if len(town_df) >= 12:
        return town_df, f"{town}, {district}"

    district_df = df[df["district"] == district].copy()
    if len(district_df) >= 12:
        return district_df, district

    return df.copy(), "all listed areas"


def style_page() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --ink: #111827;
            --muted: #5f6d80;
            --paper: #ffffff;
            --mist: #eef3f8;
            --line: #d5deea;
            --line-soft: #e8edf4;
            --teal: #009c8c;
            --teal-dark: #08786d;
            --coral: #ee6346;
            --coral-dark: #ca4b33;
            --sun: #f2b84b;
            --indigo: #4054d6;
            --navy: #141f34;
            --field: #f9fbfe;
            --sidebar: #ffffff;
        }

        html, body, .stApp {
            font-family: Manrope, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--ink);
        }

        .stApp {
            background:
                linear-gradient(180deg, rgba(255,255,255,.72) 0, rgba(255,255,255,0) 280px),
                var(--mist);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.8rem;
            max-width: 1220px;
        }

        header[data-testid="stHeader"] {
            display: none;
        }

        [data-testid="stSidebar"] {
            background: var(--sidebar);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.35rem;
        }

        [data-testid="stSidebar"] * {
            color: var(--ink);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p {
            color: var(--ink) !important;
        }

        [data-testid="stSidebar"] label,
        label {
            color: var(--ink) !important;
            font-weight: 800 !important;
            font-size: .82rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: var(--muted) !important;
        }

        .sidebar-title {
            display: flex;
            align-items: center;
            gap: .55rem;
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 800;
            margin: .15rem 0 1.25rem;
        }

        .sidebar-title::before {
            content: "";
            width: .7rem;
            height: .7rem;
            border-radius: 3px;
            background: var(--teal);
            box-shadow: 0 0 0 4px rgba(0, 156, 140, .12);
        }

        .sidebar-section {
            color: var(--muted);
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin: 1.15rem 0 .45rem;
        }

        [data-testid="stSidebar"] hr {
            border-color: var(--line-soft);
            margin: 1.1rem 0;
        }

        .top-shell {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 1rem;
            align-items: center;
            margin-bottom: 1rem;
        }

        .app-title {
            font-size: clamp(1.65rem, 2.6vw, 2.35rem);
            font-weight: 800;
            line-height: 1.08;
            margin: 0;
            color: var(--ink);
        }

        .app-subtitle {
            margin: .28rem 0 0;
            color: var(--muted);
            font-size: .92rem;
            max-width: 720px;
        }

        .header-kicker {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            color: var(--teal-dark);
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }

        .header-kicker::before {
            content: "";
            width: .65rem;
            height: .65rem;
            border-radius: 2px;
            background: var(--teal);
        }

        .model-badge {
            border: 1px solid rgba(64, 84, 214, .24);
            border-radius: 999px;
            color: var(--indigo) !important;
            background: rgba(64, 84, 214, .08);
            padding: .65rem .95rem;
            font-weight: 800;
            white-space: nowrap;
        }

        .section-heading {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 800;
            margin: 0 0 .75rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: .45rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            background: rgba(255, 255, 255, .78);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: #43516a;
            font-size: .78rem;
            font-weight: 700;
            padding: .42rem .68rem;
        }

        .panel {
            border: 1px solid var(--line);
            background: var(--paper);
            border-radius: 8px;
            padding: 1rem;
            min-height: 100%;
            box-shadow: 0 16px 38px rgba(30, 45, 68, .06);
        }

        .price-panel {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(238, 99, 70, .14), rgba(238, 99, 70, 0) 34%),
                var(--navy);
            color: white;
            border: 1px solid var(--navy);
            padding: 1.35rem;
        }

        .price-panel::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 7px;
            background: var(--coral);
        }

        .price-panel > * {
            position: relative;
            z-index: 1;
        }

        .eyebrow {
            color: var(--muted);
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }

        .price-panel .eyebrow {
            color: rgba(255, 255, 255, .82);
        }

        .price-value {
            font-size: clamp(2.35rem, 5vw, 4rem);
            font-weight: 800;
            line-height: 1;
            margin: .15rem 0 .65rem;
            color: white;
        }

        .range-text {
            color: rgba(255, 255, 255, .86);
            font-size: .95rem;
        }

        .price-meta {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: .65rem;
            margin-top: 1rem;
        }

        .price-meta-item {
            border: 1px solid rgba(255, 255, 255, .16);
            background: rgba(255, 255, 255, .08);
            border-radius: 8px;
            padding: .75rem;
        }

        .price-meta-label {
            color: rgba(255, 255, 255, .72);
            font-size: .72rem;
            font-weight: 800;
            margin-bottom: .2rem;
        }

        .price-meta-value {
            color: white;
            font-size: 1.05rem;
            font-weight: 800;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: .65rem;
        }

        .mini-metric {
            border: 1px solid var(--line);
            background: var(--paper);
            border-radius: 8px;
            padding: .85rem;
            box-shadow: 0 12px 30px rgba(30, 45, 68, .045);
        }

        .mini-label {
            color: var(--muted);
            font-size: .78rem;
            font-weight: 750;
            margin-bottom: .25rem;
        }

        .mini-value {
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .summary-table td {
            padding: .44rem 0;
            border-bottom: 1px solid #edf1f6;
        }

        .summary-table tr:last-child td {
            border-bottom: 0;
        }

        .summary-key {
            color: var(--muted);
        }

        .summary-value {
            color: var(--ink);
            text-align: right;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--line);
            background: var(--paper);
            border-radius: 8px;
            padding: .9rem 1rem;
            box-shadow: 0 12px 30px rgba(30, 45, 68, .045);
        }

        div[data-testid="stMetric"] label {
            color: var(--muted) !important;
            font-weight: 800;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--ink) !important;
            font-size: 1.75rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: .35rem;
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: .35rem;
        }

        button[data-baseweb="tab"] {
            border-radius: 6px;
            color: #475467 !important;
            font-weight: 800;
            padding-left: .9rem;
            padding-right: .9rem;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: var(--navy);
            color: white !important;
        }

        div[data-baseweb="select"] > div {
            border-radius: 8px !important;
            border-color: var(--line) !important;
            background: white !important;
            min-height: 2.55rem;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] svg,
        div[data-testid="stSelectbox"] * {
            color: var(--ink) !important;
            fill: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }

        div[data-testid="stNumberInput"] {
            color: var(--ink) !important;
        }

        div[data-testid="stNumberInput"] input {
            border-radius: 8px 0 0 8px !important;
            border-color: var(--line) !important;
            background: white !important;
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }

        div[data-testid="stNumberInput"] button {
            background: var(--navy) !important;
            border-color: var(--navy) !important;
            color: white !important;
        }

        div[data-testid="stNumberInput"] button svg {
            fill: white !important;
        }

        [data-testid="stSlider"] {
            color: var(--ink) !important;
        }

        [data-testid="stSlider"] [role="slider"] {
            background: var(--coral);
            border-color: white;
        }

        [data-testid="stSlider"] [data-testid="stTickBar"] {
            color: var(--muted) !important;
        }

        [data-testid="stSlider"] [data-baseweb="slider"] div {
            color: var(--ink) !important;
        }

        [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
            background-color: var(--coral) !important;
        }

        [data-testid="stMarkdownContainer"] p,
        .stCaptionContainer {
            color: var(--muted);
        }

        .stAlert {
            border-radius: 8px;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        @media (max-width: 760px) {
            .top-shell {
                grid-template-columns: 1fr;
            }

            .metric-grid,
            .price-meta {
                grid-template-columns: 1fr;
            }
        }

        footer, #MainMenu, .stDeployButton {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(meta: dict[str, Any]) -> None:
    params = meta.get("best_params", {})
    model_label = "CatBoost"
    if meta.get("model_type") == "sklearn_hgb_onehot":
        model_label = "HistGradientBoosting"

    st.markdown(
        f"""
        <div class="top-shell">
            <div>
                <div class="header-kicker">Live valuation studio</div>
                <h1 class="app-title">Sri Lanka House Price Predictor</h1>
                <p class="app-subtitle">Property valuation, comparable listings, and model diagnostics in one workspace.</p>
            </div>
            <div class="model-badge">{model_label} model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if params:
        st.markdown(
            f"""
            <div class="chip-row" style="margin-bottom: .95rem;">
                <div class="chip">{meta.get('splits', {}).get('train', '-')} train rows</div>
                <div class="chip">{meta.get('splits', {}).get('test', '-')} test rows</div>
                <div class="chip">{params.get('iterations', '-')} boosting iterations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def sidebar_inputs(meta: dict[str, Any], town_map: dict[str, list[str]]) -> dict[str, Any]:
    categories = meta.get("categories", {})
    inputs: dict[str, Any] = {}

    with st.sidebar:
        st.markdown("<div class='sidebar-title'>Valuation inputs</div>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-section'>Location</div>", unsafe_allow_html=True)
        districts = categories.get("district", [])
        default_district = districts.index("Colombo") if "Colombo" in districts else 0
        district = st.selectbox("District", options=districts, index=default_district)
        inputs["district"] = district

        towns = town_map.get(district) or categories.get("town", [])
        town = st.selectbox("Town", options=towns, index=0)
        inputs["town"] = town

        st.divider()
        st.markdown("<div class='sidebar-section'>Size</div>", unsafe_allow_html=True)
        inputs["Land size"] = st.number_input(
            "Land size (perches)",
            min_value=1.0,
            max_value=500.0,
            value=10.0,
            step=1.0,
        )
        inputs["House size"] = st.number_input(
            "House size (sq ft)",
            min_value=100.0,
            max_value=20_000.0,
            value=1_500.0,
            step=50.0,
        )

        st.divider()
        st.markdown("<div class='sidebar-section'>Rooms</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            inputs["Beds"] = st.slider("Beds", 1, 12, 3)
        with col2:
            inputs["Baths"] = st.slider("Baths", 1, 10, 2)

        seller_options = categories.get("Seller_type", ["Member", "Premium-Member"])
        default_seller = seller_options.index("Member") if "Member" in seller_options else 0
        inputs["Seller_type"] = st.selectbox("Seller type", options=seller_options, index=default_seller)

        st.divider()
        st.caption("Changing any field recalculates the estimate instantly.")

    return inputs


def main_inputs(meta: dict[str, Any], town_map: dict[str, list[str]]) -> dict[str, Any]:
    categories = meta.get("categories", {})
    inputs: dict[str, Any] = {}

    st.markdown("<div class='section-heading'>Valuation inputs</div>", unsafe_allow_html=True)
    location_col, size_col, room_col = st.columns([1.05, 1, .95], gap="medium")

    with location_col:
        districts = categories.get("district", [])
        default_district = districts.index("Colombo") if "Colombo" in districts else 0
        district = st.selectbox("District", options=districts, index=default_district)
        inputs["district"] = district

        towns = town_map.get(district) or categories.get("town", [])
        inputs["town"] = st.selectbox("Town", options=towns, index=0)

    with size_col:
        inputs["Land size"] = st.number_input(
            "Land size (perches)",
            min_value=1.0,
            max_value=500.0,
            value=10.0,
            step=1.0,
        )
        inputs["House size"] = st.number_input(
            "House size (sq ft)",
            min_value=100.0,
            max_value=20_000.0,
            value=1_500.0,
            step=50.0,
        )

    with room_col:
        inputs["Beds"] = st.slider("Beds", 1, 12, 3)
        inputs["Baths"] = st.slider("Baths", 1, 10, 2)
        seller_options = categories.get("Seller_type", ["Member", "Premium-Member"])
        default_seller = seller_options.index("Member") if "Member" in seller_options else 0
        inputs["Seller_type"] = st.selectbox("Seller type", options=seller_options, index=default_seller)

    return inputs


def render_price_panel(prediction: PredictionResult) -> None:
    st.markdown(
        f"""
        <div class="panel price-panel">
            <div class="eyebrow">Estimated market value</div>
            <div class="price-value">{format_lkr(prediction.value)}</div>
            <div class="range-text">
                Expected range: {format_lkr(prediction.lower)} to {format_lkr(prediction.upper)}
            </div>
            <div class="price-meta">
                <div class="price-meta-item">
                    <div class="price-meta-label">Lower guide</div>
                    <div class="price-meta-value">{format_lkr(prediction.lower)}</div>
                </div>
                <div class="price-meta-item">
                    <div class="price-meta-label">Upper guide</div>
                    <div class="price-meta-value">{format_lkr(prediction.upper)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_property_summary(inputs: dict[str, Any]) -> None:
    rows = [
        ("District", inputs["district"]),
        ("Town", inputs["town"]),
        ("Beds", inputs["Beds"]),
        ("Baths", inputs["Baths"]),
        ("Land", f"{inputs['Land size']:,.0f} perches"),
        ("House", f"{inputs['House size']:,.0f} sq ft"),
        ("Seller", inputs["Seller_type"]),
    ]
    table_rows = "".join(
        f"<tr><td class='summary-key'>{key}</td><td class='summary-value'>{value}</td></tr>"
        for key, value in rows
    )
    st.markdown(
        f"""
        <div class="panel">
            <div class="eyebrow">Current property</div>
            <table class="summary-table" style="width:100%; border-collapse:collapse;">
                {table_rows}
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_market_cards(df: pd.DataFrame, label: str, prediction: PredictionResult) -> None:
    prices = df["Price"].dropna()
    median = float(prices.median()) if len(prices) else 0.0
    q25 = float(prices.quantile(0.25)) if len(prices) else 0.0
    q75 = float(prices.quantile(0.75)) if len(prices) else 0.0
    delta_pct = ((prediction.value - median) / median * 100.0) if median else 0.0

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="mini-metric">
                <div class="mini-label">Comparable area</div>
                <div class="mini-value">{label}</div>
            </div>
            <div class="mini-metric">
                <div class="mini-label">Comparable listings</div>
                <div class="mini-value">{len(df):,}</div>
            </div>
            <div class="mini-metric">
                <div class="mini-label">Median listed price</div>
                <div class="mini-value">{format_lkr(median)}</div>
            </div>
            <div class="mini-metric">
                <div class="mini-label">Middle 50 percent</div>
                <div class="mini-value">{format_lkr(q25, decimals=0)} - {format_lkr(q75, decimals=0)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"The model estimate is {delta_pct:+.1f}% versus the comparable-area median.")


def dark_axis(figsize: tuple[float, float] = (7, 4)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#667085", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#dbe3ed")
    ax.grid(axis="x", color="#edf2f7", linewidth=.8)
    ax.set_axisbelow(True)
    return fig, ax


def price_distribution_chart(df: pd.DataFrame, prediction: PredictionResult, label: str) -> plt.Figure:
    fig, ax = dark_axis((8, 4.2))
    prices_m = df["Price"].dropna().clip(lower=0) / 1_000_000
    upper = max(float(prices_m.quantile(0.97)), prediction.value / 1_000_000 * 1.15, 1)
    shown = prices_m[prices_m <= upper]

    ax.hist(shown, bins=36, color="#bde7ee", edgecolor="#ffffff", density=False)
    ax.axvline(prediction.value / 1_000_000, color="#ff6b4a", lw=2.8, label="Estimate")
    ax.axvline(float(prices_m.median()), color="#4b5cff", lw=1.8, ls="--", label="Median")
    ax.set_title(f"Price distribution: {label}", color="#14161f", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Price (million LKR)", color="#667085")
    ax.set_ylabel("Listings", color="#667085")
    ax.legend(frameon=False, labelcolor="#14161f")
    fig.tight_layout()
    return fig


def contribution_chart(contrib: pd.DataFrame) -> plt.Figure:
    top = contrib.head(8).sort_values("contribution")
    colors = ["#ff6b4a" if value < 0 else "#00a6a6" for value in top["contribution"]]

    fig, ax = dark_axis((8, 4.3))
    ax.barh(top["feature"], top["contribution"] / 1_000_000, color=colors)
    ax.axvline(0, color="#98a2b3", lw=1)
    ax.set_title("Top drivers for this estimate", color="#14161f", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Impact (million LKR)", color="#667085")
    fig.tight_layout()
    return fig


def global_importance_chart(model: Any, feature_columns: list[str]) -> plt.Figure:
    fi = np.array(model.get_feature_importance(type="FeatureImportance"), dtype=float)
    order = np.argsort(fi)[::-1][:10]
    names = [feature_columns[i] for i in order][::-1]
    values = [fi[i] for i in order][::-1]

    fig, ax = dark_axis((8, 4.3))
    bar_colors = ["#00a6a6", "#23b8b8", "#4b5cff", "#6675ff", "#ff6b4a"] * 2
    ax.barh(names, values, color=bar_colors[: len(values)])
    ax.set_title("Global feature importance", color="#14161f", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Importance score", color="#667085")
    fig.tight_layout()
    return fig


def actual_vs_predicted_chart() -> plt.Figure | None:
    true_path = ARTIFACTS_DIR / "test_true.npy"
    pred_path = ARTIFACTS_DIR / "test_pred.npy"
    if not true_path.exists() or not pred_path.exists():
        return None

    y_true = np.load(true_path)
    y_pred = np.load(pred_path)
    fig, ax = dark_axis((6.8, 4.5))
    ax.scatter(y_true, y_pred, color="#00a6a6", s=12, alpha=0.38, edgecolors="none")

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="#ff6b4a", lw=1.6, ls="--")
    ax.set_title("Actual vs predicted", color="#14161f", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Actual price", color="#667085")
    ax.set_ylabel("Predicted price", color="#667085")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
    fig.tight_layout()
    return fig


def render_quality(meta: dict[str, Any]) -> None:
    metrics = meta.get("test_metrics", {})
    r2 = float(metrics.get("r2", 0.0) or 0.0)
    rmse = float(metrics.get("rmse", 0.0) or 0.0)
    mae = float(metrics.get("mae", 0.0) or 0.0)

    col1, col2, col3 = st.columns(3)
    col1.metric("R2 score", f"{r2:.3f}")
    col2.metric("MAE", format_lkr(mae))
    col3.metric("RMSE", format_lkr(rmse))


def main() -> None:
    st.set_page_config(
        page_title="Sri Lanka House Price Predictor",
        page_icon="house",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    style_page()

    model, meta = load_model_and_meta(ARTIFACTS_DIR)
    town_map = district_town_map(DATASET_PATH, meta.get("categories", {}))
    market_df = load_market_data(DATASET_PATH)

    render_header(meta)

    inputs = sidebar_inputs(meta, town_map)
    feature_cols = meta["feature_columns"]
    cat_cols = meta.get("categorical_columns", [])
    x_row = inputs_to_dataframe(inputs, feature_cols, cat_cols)
    prediction = predict_price(model, meta, x_row)
    comparable_df, comparable_label = market_summary(market_df, inputs["district"], inputs["town"])

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        render_price_panel(prediction)
    with right:
        render_property_summary(inputs)

    st.write("")

    tab_overview, tab_explain, tab_model = st.tabs(["Overview", "Explanation", "Model quality"])

    with tab_overview:
        col1, col2 = st.columns([0.9, 1.1], gap="large")
        with col1:
            render_market_cards(comparable_df, comparable_label, prediction)
        with col2:
            st.pyplot(price_distribution_chart(comparable_df, prediction, comparable_label), clear_figure=True)

    with tab_explain:
        if prediction.model_type == "sklearn_hgb_onehot":
            st.info("Local contribution charts are available for the CatBoost artifact.")
        else:
            contrib = local_contributions(model, x_row, feature_cols, cat_cols)
            col1, col2 = st.columns([1.15, 0.85], gap="large")
            with col1:
                st.pyplot(contribution_chart(contrib), clear_figure=True)
            with col2:
                shown = contrib.head(7).copy()
                shown["impact"] = shown["contribution"].map(lambda v: format_lkr(abs(v)))
                shown["direction"] = np.where(shown["contribution"] >= 0, "Raises", "Lowers")
                st.dataframe(
                    shown[["feature", "direction", "impact"]],
                    use_container_width=True,
                    hide_index=True,
                )

    with tab_model:
        render_quality(meta)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            if prediction.model_type != "sklearn_hgb_onehot":
                st.pyplot(global_importance_chart(model, feature_cols), clear_figure=True)
        with col2:
            avp_fig = actual_vs_predicted_chart()
            if avp_fig is not None:
                st.pyplot(avp_fig, clear_figure=True)
            else:
                st.info("Test prediction arrays were not found in artifacts.")


if __name__ == "__main__":
    main()
