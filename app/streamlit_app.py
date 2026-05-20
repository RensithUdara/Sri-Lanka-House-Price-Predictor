from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
import joblib

try:
    import shap
except Exception:
    shap = None

try:
    from catboost import CatBoostRegressor, Pool
except Exception:
    CatBoostRegressor = None
    Pool = None

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

# ── Cached helpers ──────────────────────────────────────────────────────
_META_CACHE: dict | None = None


def _load_meta() -> dict:
    global _META_CACHE
    if _META_CACHE is None:
        _META_CACHE = json.loads(
            (ARTIFACTS_DIR / "model_meta.json").read_text(encoding="utf-8")
        )
    return _META_CACHE


def load_model_and_meta(artifacts_dir: Path):
    meta = _load_meta()
    sklearn_path = artifacts_dir / "hgb_house_price.joblib"
    if sklearn_path.exists():
        model = joblib.load(sklearn_path)
        return model, meta
    if CatBoostRegressor is None:
        raise RuntimeError("No model artifact found.")
    model = CatBoostRegressor()
    model.load_model(artifacts_dir / "catboost_house_price.cbm")
    return model, meta


# ── CSS injection ───────────────────────────────────────────────────────
def _inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* hero banner */
        .hero {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 40%, #2c5364 100%);
            border-radius: 18px;
            padding: 2.8rem 2.2rem 2.2rem;
            margin-bottom: 1.6rem;
            color: white;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,.25);
        }
        .hero h1 { font-size: 2.3rem; font-weight: 800; margin: 0 0 .3rem; letter-spacing: -.5px; }
        .hero p  { font-size: 1.05rem; opacity: .85; margin: 0; }

        /* glass card */
        .glass-card {
            background: rgba(255,255,255,.06);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 16px;
            padding: 1.6rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 24px rgba(0,0,0,.08);
        }
        .glass-card h3 { margin-top: 0; }

        /* price card */
        .price-card {
            background: linear-gradient(135deg, #00b09b, #96c93d);
            border-radius: 18px;
            padding: 2.2rem 1.4rem;
            text-align: center;
            color: white;
            box-shadow: 0 6px 28px rgba(0,176,155,.35);
            margin-bottom: .8rem;
        }
        .price-card .label { font-size: .95rem; opacity: .9; margin-bottom: .3rem; }
        .price-card .value { font-size: 2.6rem; font-weight: 800; letter-spacing: -1px; }
        .price-card .sub   { font-size: .82rem; opacity: .75; margin-top: .5rem; }

        /* stat pills row */
        .stat-row { display: flex; gap: .7rem; flex-wrap: wrap; margin-top: .6rem; }
        .stat-pill {
            flex: 1 1 120px;
            background: rgba(255,255,255,.07);
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 12px;
            padding: .9rem .8rem;
            text-align: center;
        }
        .stat-pill .stat-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .6px; opacity: .65; }
        .stat-pill .stat-value { font-size: 1.15rem; font-weight: 700; margin-top: .15rem; }

        /* section titles */
        .section-title {
            font-size: 1.15rem; font-weight: 700;
            margin: 1.4rem 0 .6rem;
            display: flex; align-items: center; gap: .45rem;
        }

        /* hide streamlit chrome */
        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }
        header[data-testid="stHeader"] button[kind="header"] { display: none; }
        .stDeployButton { display: none !important; }

        /* nicer form controls */
        div[data-baseweb="select"] > div { border-radius: 10px !important; }
        input[type="number"] { border-radius: 10px !important; }
        button[data-baseweb="tab"] { font-weight: 600; }

        /* sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f2027 0%, #203a43 100%);
        }
        section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
        section[data-testid="stSidebar"] label { font-weight: 600 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero():
    st.markdown(
        """
        <div class="hero">
            <h1>🏠 Sri Lanka House Price Predictor</h1>
            <p>AI-powered property valuation · gradient-boosted trees · 15 000+ listings</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _price_card(pred: float):
    if pred >= 1_000_000:
        display = f"LKR {pred / 1_000_000:,.2f} M"
    else:
        display = f"LKR {pred:,.0f}"
    st.markdown(
        f"""
        <div class="price-card">
            <div class="label">Estimated Market Value</div>
            <div class="value">{display}</div>
            <div class="sub">Point estimate · actual price may vary</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _model_quality_pills(meta: dict):
    tm = meta.get("test_metrics", {})
    r2 = round(float(tm.get("r2", 0)) * 100, 1)
    rmse_m = round(float(tm.get("rmse", 0)) / 1_000_000, 1)
    mae_m = round(float(tm.get("mae", 0)) / 1_000_000, 1)
    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-pill">
                <div class="stat-label">Accuracy (R²)</div>
                <div class="stat-value">{r2}%</div>
            </div>
            <div class="stat-pill">
                <div class="stat-label">Avg Error (MAE)</div>
                <div class="stat-value">{mae_m} M</div>
            </div>
            <div class="stat-pill">
                <div class="stat-label">RMSE</div>
                <div class="stat-value">{rmse_m} M</div>
            </div>
            <div class="stat-pill">
                <div class="stat-label">Test Samples</div>
                <div class="stat-value">{meta.get("splits", {}).get("test", "—")}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── District → Town mapping (from CSV) ─────────────────────────────────
@st.cache_data(show_spinner=False)
def _district_town_map() -> dict[str, list[str]]:
    csv_path = Path(__file__).resolve().parents[1] / "dataset" / "cleaned_data_house_prices.csv"
    try:
        import csv as _csv, sys as _sys
        try:
            _csv.field_size_limit(min(_sys.maxsize, 2**31 - 1))
        except Exception:
            pass
        df = pd.read_csv(csv_path, sep="\t", engine="python", usecols=["district", "town"])
        df["district"] = df["district"].astype(str).str.strip()
        df["town"] = df["town"].astype(str).str.strip()
        mapping: dict[str, list[str]] = {}
        for d, g in df.groupby("district"):
            mapping[d] = sorted(g["town"].dropna().unique().tolist())
        return mapping
    except Exception:
        meta = _load_meta()
        towns = meta.get("categories", {}).get("town", [])
        districts = meta.get("categories", {}).get("district", [])
        return {d: towns for d in districts}


# ── Sidebar inputs ──────────────────────────────────────────────────────
def _sidebar_inputs(meta: dict) -> dict:
    inputs: dict = {}

    with st.sidebar:
        st.markdown("### 🏡 Property Details")
        st.caption("Adjust the details — prediction updates live.")

        st.markdown("---")

        # Location
        st.markdown("**📍 Location**")
        districts = meta.get("categories", {}).get("district", [])
        district = st.selectbox("District", options=districts, index=0, help="Select the district")
        inputs["district"] = district

        dt_map = _district_town_map()
        available_towns = dt_map.get(district, meta.get("categories", {}).get("town", []))
        town = st.selectbox("Town", options=available_towns, index=0, help="Towns in the selected district")
        inputs["town"] = town

        st.markdown("---")

        # Size
        st.markdown("**📐 Size**")
        c1, c2 = st.columns(2)
        with c1:
            inputs["Land size"] = st.number_input(
                "Land (perches)", min_value=0.0, max_value=500.0, value=10.0, step=1.0
            )
        with c2:
            inputs["House size"] = st.number_input(
                "House (sqft)", min_value=0.0, max_value=20000.0, value=1500.0, step=100.0
            )

        st.markdown("---")

        # Rooms
        st.markdown("**🛏️ Rooms**")
        c1, c2 = st.columns(2)
        with c1:
            inputs["Beds"] = st.slider("Bedrooms", 1, 12, 3)
        with c2:
            inputs["Baths"] = st.slider("Bathrooms", 1, 10, 2)

        # Hidden default (model still needs Seller_type)
        inputs["Seller_type"] = "Member"

        st.markdown("---")
        st.caption("Model: CatBoost · 15 269 listings · R² ≈ 72 %")

    return inputs


# ── DataFrame builder ───────────────────────────────────────────────────
def inputs_to_dataframe(
    inputs: dict,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    row: dict = {}
    for col in feature_columns:
        if col in categorical_columns:
            row[col] = str(inputs.get(col, "__MISSING__") or "__MISSING__").strip()
        else:
            try:
                row[col] = float(inputs.get(col, 0.0))
            except Exception:
                row[col] = np.nan
    return pd.DataFrame([row], columns=feature_columns)


# ── Charts ──────────────────────────────────────────────────────────────
def _shap_chart(model, x_row, cat_cols, feature_cols):
    cat_idx = [feature_cols.index(c) for c in cat_cols]
    pool = Pool(x_row, cat_features=cat_idx)
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    contrib = np.array(shap_values[0][:-1], dtype=float)
    order = np.argsort(np.abs(contrib))[::-1][:8]

    names = [feature_cols[i] for i in order][::-1]
    vals = [contrib[i] for i in order][::-1]
    colors = ["#00b09b" if v >= 0 else "#e74c3c" for v in vals]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.barh(names, vals, color=colors, height=0.6, edgecolor="none")
    ax.axvline(0, color="#555", lw=0.8)
    ax.tick_params(colors="#ccc", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#444")
    ax.set_xlabel("Impact on prediction", color="#aaa", fontsize=9)
    ax.set_title("What's driving this price?", color="white", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def _importance_chart(model, feature_cols):
    fi = model.get_feature_importance(type="FeatureImportance")
    order = np.argsort(fi)[::-1][:10]
    names = [feature_cols[i] for i in order][::-1]
    vals = [fi[i] for i in order][::-1]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.barh(names, vals, color="#96c93d", height=0.6, edgecolor="none")
    ax.tick_params(colors="#ccc", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#444")
    ax.set_xlabel("Importance score", color="#aaa", fontsize=9)
    ax.set_title("Feature Importance (global)", color="white", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


# ── Load dataset sample + SHAP values (cached) ─────────────────────────
@st.cache_data(show_spinner="Computing SHAP values on dataset sample…")
def _load_sample_shap(_model_id: str, n: int = 500):
    """Return (sample_df, shap_array, expected_value, feature_cols, cat_cols)."""
    from src.data import DEFAULT_SCHEMA, infer_feature_types, load_dataset

    meta = _load_meta()
    feature_cols = meta["feature_columns"]
    cat_cols = meta.get("categorical_columns", [])

    csv_path = Path(__file__).resolve().parents[1] / "dataset" / "cleaned_data_house_prices.csv"
    df = load_dataset(str(csv_path), schema=DEFAULT_SCHEMA)

    # Deterministic sample
    sample = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)

    x = sample[feature_cols].copy()
    for c in cat_cols:
        x[c] = x[c].astype(str).fillna("__MISSING__")

    cat_idx = [feature_cols.index(c) for c in cat_cols]
    pool = Pool(x, cat_features=cat_idx)

    model = CatBoostRegressor()
    model.load_model(ARTIFACTS_DIR / "catboost_house_price.cbm")

    shap_raw = model.get_feature_importance(pool, type="ShapValues")
    shap_vals = np.array(shap_raw[:, :-1], dtype=float)   # (n, features)
    expected = float(shap_raw[0, -1])

    return sample, shap_vals, expected, feature_cols, cat_cols, x


def _dark_ax(figsize=(8, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="#ccc", labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color("#444")
    return fig, ax


# ── SHAP Waterfall (single prediction) ─────────────────────────────────
def _waterfall_chart(model, x_row, cat_cols, feature_cols, pred):
    cat_idx = [feature_cols.index(c) for c in cat_cols]
    pool = Pool(x_row, cat_features=cat_idx)
    sv = model.get_feature_importance(pool, type="ShapValues")
    contrib = np.array(sv[0][:-1], dtype=float)
    base = float(sv[0][-1])
    order = np.argsort(np.abs(contrib))[::-1][:7]

    labels = [feature_cols[i] for i in order]
    values = [contrib[i] for i in order]

    fig, ax = _dark_ax((8, 5))
    cumulative = base
    y_pos = list(range(len(labels) + 1))[::-1]

    # Base value bar
    ax.barh(y_pos[0], base, color="#555", height=0.5, left=0)
    ax.text(base / 2, y_pos[0], f"Base\n{base / 1e6:.1f}M", ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    for i, (lbl, val) in enumerate(zip(labels, values)):
        color = "#00b09b" if val >= 0 else "#e74c3c"
        ax.barh(y_pos[i + 1], val, left=cumulative, color=color, height=0.5, edgecolor="none")
        text_x = cumulative + val / 2
        sign = "+" if val >= 0 else ""
        ax.text(text_x, y_pos[i + 1], f"{sign}{val / 1e6:.1f}M", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        cumulative += val

    all_labels = ["Base Value"] + labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_labels, color="#ccc", fontsize=9)
    ax.axvline(base, color="#666", ls="--", lw=0.7)
    ax.axvline(pred, color="#96c93d", ls="--", lw=1.2, label=f"Prediction: {pred / 1e6:.1f}M")
    ax.set_xlabel("Price (LKR)", color="#aaa", fontsize=9)
    ax.set_title("SHAP Waterfall — How the prediction is built", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8, facecolor="#0e1117", edgecolor="#444", labelcolor="white")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
    fig.tight_layout()
    return fig


# ── SHAP Bee-swarm style scatter (global) ───────────────────────────────
def _shap_beeswarm(shap_vals, x_df, feature_cols):
    """Horizontal strip plot: one dot per sample, colored by feature value."""
    num_cols = [c for c in feature_cols if c in ["Baths", "Land size", "Beds", "House size"]]
    fig, axes = plt.subplots(len(num_cols), 1, figsize=(8, 2.6 * len(num_cols)), sharex=False)
    fig.patch.set_facecolor("#0e1117")
    if len(num_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, num_cols):
        idx = feature_cols.index(col)
        sv = shap_vals[:, idx]
        fv = pd.to_numeric(x_df[col], errors="coerce").fillna(0).values
        ax.set_facecolor("#0e1117")

        norm = plt.Normalize(vmin=np.nanpercentile(fv, 5), vmax=np.nanpercentile(fv, 95))
        colors = plt.cm.coolwarm(norm(fv))

        # Add jitter on y
        jitter = np.random.default_rng(42).normal(0, 0.12, size=len(sv))
        ax.scatter(sv, jitter, c=colors, s=12, alpha=0.6, edgecolors="none")
        ax.axvline(0, color="#555", lw=0.7)
        ax.set_ylabel(col, color="#ccc", fontsize=10, fontweight="bold")
        ax.set_yticks([])
        ax.tick_params(colors="#ccc", labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("bottom", "left"):
            ax.spines[s].set_color("#444")

        # Color bar
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
        cb.set_label("Feature value", color="#aaa", fontsize=7)
        cb.ax.tick_params(colors="#ccc", labelsize=7)

    axes[0].set_title("SHAP Scatter — Feature value vs. Impact", color="white", fontsize=12, fontweight="bold", pad=10)
    axes[-1].set_xlabel("SHAP value (impact on price)", color="#aaa", fontsize=9)
    fig.tight_layout()
    return fig


# ── SHAP Dependence Plot (scatter: feature value vs SHAP value) ─────────
def _shap_dependence(shap_vals, x_df, feature_cols, col_name):
    idx = feature_cols.index(col_name)
    sv = shap_vals[:, idx]
    fv = pd.to_numeric(x_df[col_name], errors="coerce").fillna(0).values

    fig, ax = _dark_ax((7, 4.5))
    ax.scatter(fv, sv, c="#00b09b", s=14, alpha=0.5, edgecolors="none")

    # Trend line
    if len(fv) > 10:
        z = np.polyfit(fv, sv, 2)
        p = np.poly1d(z)
        x_line = np.linspace(np.nanpercentile(fv, 2), np.nanpercentile(fv, 98), 100)
        ax.plot(x_line, p(x_line), color="#e74c3c", lw=2, ls="--", label="Trend")
        ax.legend(fontsize=8, facecolor="#0e1117", edgecolor="#444", labelcolor="white")

    ax.axhline(0, color="#555", lw=0.7)
    ax.set_xlabel(col_name, color="#aaa", fontsize=10)
    ax.set_ylabel("SHAP value", color="#aaa", fontsize=10)
    ax.set_title(f"Dependence Plot — {col_name}", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M" if abs(x) >= 1e6 else f"{x / 1e3:.0f}K"))
    fig.tight_layout()
    return fig


# ── Actual vs Predicted scatter ─────────────────────────────────────────
def _actual_vs_pred_scatter():
    test_true_path = ARTIFACTS_DIR / "test_true.npy"
    test_pred_path = ARTIFACTS_DIR / "test_pred.npy"
    if not test_true_path.exists() or not test_pred_path.exists():
        return None
    y_true = np.load(test_true_path)
    y_pred = np.load(test_pred_path)

    fig, ax = _dark_ax((7, 5))
    ax.scatter(y_true, y_pred, c="#00b09b", s=10, alpha=0.35, edgecolors="none")

    # Perfect line
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="#e74c3c", lw=1.5, ls="--", label="Perfect prediction")

    ax.set_xlabel("Actual Price (LKR)", color="#aaa", fontsize=10)
    ax.set_ylabel("Predicted Price (LKR)", color="#aaa", fontsize=10)
    ax.set_title("Actual vs Predicted — Test Set", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=8, facecolor="#0e1117", edgecolor="#444", labelcolor="white")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
    fig.tight_layout()
    return fig


# ── Residual line chart ─────────────────────────────────────────────────
def _residual_chart():
    test_true_path = ARTIFACTS_DIR / "test_true.npy"
    test_pred_path = ARTIFACTS_DIR / "test_pred.npy"
    if not test_true_path.exists() or not test_pred_path.exists():
        return None
    y_true = np.load(test_true_path)
    y_pred = np.load(test_pred_path)

    residuals = y_true - y_pred
    # Sort by actual for smooth line
    order = np.argsort(y_true)
    y_sorted = y_true[order]
    r_sorted = residuals[order]

    fig, ax = _dark_ax((8, 4))
    ax.scatter(y_sorted, r_sorted, c="#96c93d", s=8, alpha=0.3, edgecolors="none")
    ax.axhline(0, color="#e74c3c", lw=1.5, ls="--")

    # Moving average line
    window = max(len(r_sorted) // 30, 10)
    if len(r_sorted) > window:
        ma = np.convolve(r_sorted, np.ones(window) / window, mode="valid")
        ax.plot(y_sorted[:len(ma)], ma, color="#00b09b", lw=2, label=f"Moving avg (w={window})")
        ax.legend(fontsize=8, facecolor="#0e1117", edgecolor="#444", labelcolor="white")

    ax.set_xlabel("Actual Price (LKR)", color="#aaa", fontsize=10)
    ax.set_ylabel("Residual (Actual − Predicted)", color="#aaa", fontsize=10)
    ax.set_title("Residuals — Where the model over/under-predicts", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
    fig.tight_layout()
    return fig


# ── Price distribution line chart ───────────────────────────────────────
def _price_distribution_chart(pred_value: float):
    test_true_path = ARTIFACTS_DIR / "test_true.npy"
    if not test_true_path.exists():
        return None
    y_true = np.load(test_true_path)

    fig, ax = _dark_ax((7, 3.5))
    # KDE-style histogram
    ax.hist(y_true / 1e6, bins=60, color="#203a43", edgecolor="#2c5364", alpha=0.8, density=True)
    ax.axvline(pred_value / 1e6, color="#00b09b", lw=2.5, ls="-", label=f"Your prediction: {pred_value / 1e6:.1f}M")
    ax.axvline(np.median(y_true) / 1e6, color="#e74c3c", lw=1.5, ls="--", label=f"Median: {np.median(y_true) / 1e6:.1f}M")
    ax.set_xlabel("Price (Millions LKR)", color="#aaa", fontsize=10)
    ax.set_ylabel("Density", color="#aaa", fontsize=10)
    ax.set_title("Where does your property sit?", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, np.percentile(y_true / 1e6, 97))
    ax.legend(fontsize=8, facecolor="#0e1117", edgecolor="#444", labelcolor="white")
    fig.tight_layout()
    return fig
def main() -> None:
    st.set_page_config(
        page_title="🏠 House Price Predictor",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    _hero()

    model, meta = load_model_and_meta(ARTIFACTS_DIR)
    inputs = _sidebar_inputs(meta)

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

    # ── Two tabs: Result & Explanation ────────────────
    tab_result, tab_explain = st.tabs(["🏠 Result", "🧠 Explanation"])

    with tab_result:
        st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)
        _price_card(pred)
        st.markdown(
            f"""
            <div class="glass-card">
                <h3 style="margin:0 0 .6rem;">📋 Property Summary</h3>
                <table style="width:100%; font-size:.92rem;">
                    <tr><td style="opacity:.6">District</td><td style="text-align:right;font-weight:600">{inputs["district"]}</td></tr>
                    <tr><td style="opacity:.6">Town</td><td style="text-align:right;font-weight:600">{inputs["town"]}</td></tr>
                    <tr><td style="opacity:.6">Bedrooms</td><td style="text-align:right;font-weight:600">{inputs["Beds"]}</td></tr>
                    <tr><td style="opacity:.6">Bathrooms</td><td style="text-align:right;font-weight:600">{inputs["Baths"]}</td></tr>
                    <tr><td style="opacity:.6">Land Size</td><td style="text-align:right;font-weight:600">{inputs["Land size"]} perches</td></tr>
                    <tr><td style="opacity:.6">House Size</td><td style="text-align:right;font-weight:600">{inputs["House size"]:,.0f} sqft</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        dist_fig = _price_distribution_chart(pred)
        if dist_fig:
            st.pyplot(dist_fig, clear_figure=True)

    with tab_explain:
        st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)
        _model_quality_pills(meta)
        st.markdown("")
        if model_type != "sklearn_hgb_onehot":
            fig = _shap_chart(model, x_row, cat_cols, feature_cols)
            st.pyplot(fig, clear_figure=True)
            wf_fig = _waterfall_chart(model, x_row, cat_cols, feature_cols, pred)
            st.pyplot(wf_fig, clear_figure=True)
        # Deep-dive tabs inside explanation
        st.markdown("---")
        subtab_scatter, subtab_dep, subtab_avp, subtab_imp, subtab_about = st.tabs([
            "🔵 SHAP Scatter",
            "📈 Dependence Plots",
            "🎯 Actual vs Predicted",
            "🔍 Feature Importance",
            "ℹ️ About",
        ])
        with subtab_scatter:
            st.subheader("SHAP Scatter Plots")
            st.caption("Each dot = one property from a 500-sample subset. Color = feature value. Position = SHAP impact on price.")
            if model_type != "sklearn_hgb_onehot":
                sample_df, shap_vals, expected, fc, cc, x_df = _load_sample_shap("catboost")
                bee_fig = _shap_beeswarm(shap_vals, x_df, fc)
                st.pyplot(bee_fig, clear_figure=True)
            else:
                st.info("SHAP scatter plots are available for CatBoost models.")
        with subtab_dep:
            st.subheader("SHAP Dependence Plots")
            st.caption("How does changing one feature affect price? Each dot is a property. The trend line shows the average effect.")
            if model_type != "sklearn_hgb_onehot":
                sample_df, shap_vals, expected, fc, cc, x_df = _load_sample_shap("catboost")
                num_features = [c for c in fc if c in ["Baths", "Land size", "Beds", "House size"]]
                selected_feat = st.selectbox("Select feature", options=num_features, index=num_features.index("House size") if "House size" in num_features else 0)
                dep_fig = _shap_dependence(shap_vals, x_df, fc, selected_feat)
                st.pyplot(dep_fig, clear_figure=True)
                st.markdown("#### All Numeric Features")
                c1, c2 = st.columns(2)
                for i, feat in enumerate(num_features):
                    with (c1 if i % 2 == 0 else c2):
                        dep_fig = _shap_dependence(shap_vals, x_df, fc, feat)
                        st.pyplot(dep_fig, clear_figure=True)
            else:
                st.info("Dependence plots are available for CatBoost models.")
        with subtab_avp:
            st.subheader("Actual vs Predicted Prices")
            st.caption("How well does the model predict? Points close to the red dashed line = accurate predictions.")
            col1, col2 = st.columns(2)
            with col1:
                avp_fig = _actual_vs_pred_scatter()
                if avp_fig:
                    st.pyplot(avp_fig, clear_figure=True)
                else:
                    st.warning("Test predictions not found. Run the full training pipeline to generate them.")
            with col2:
                res_fig = _residual_chart()
                if res_fig:
                    st.pyplot(res_fig, clear_figure=True)
                else:
                    st.warning("Test predictions not found.")
        with subtab_imp:
            if model_type != "sklearn_hgb_onehot":
                fig = _importance_chart(model, feature_cols)
                st.pyplot(fig, clear_figure=True)
                st.caption("This shows how much each feature contributes to the model's decisions overall (across all predictions).")
            else:
                st.info("Global importance chart available for CatBoost models.")
        with subtab_about:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    """
                    **Model Details**
                    - **Algorithm:** CatBoost (Gradient Boosted Trees)
                    - **Iterations:** 700 · **Depth:** 8
                    - **Learning Rate:** 0.08
                    - **Training Samples:** 10 687
                    """
                )
            with c2:
                st.markdown(
                    """
                    **Dataset**
                    - **Source:** Sri Lankan property listings
                    - **Total Records:** 15 269
                    - **Features:** 7 (4 numeric + 3 categorical)
                    - **Target:** Sale price (LKR)
                    """
                )
            st.markdown("---")
            st.markdown(
                """
                **SHAP Explanation Guide**
                - **Bar chart** (top right): Shows which features push your specific prediction up or down
                - **Waterfall**: Shows how the prediction is built step-by-step from the base value
                - **Scatter plots**: Each dot = one property; shows how feature values correlate with SHAP impact
                - **Dependence plots**: Reveals non-linear relationships — how changing a feature affects price
                - **Actual vs Predicted**: Quality check — how close the model's predictions are to real prices
                - **Residuals**: Where the model struggles — over-predicts vs under-predicts
                """
            )

if __name__ == "__main__":
    main()
