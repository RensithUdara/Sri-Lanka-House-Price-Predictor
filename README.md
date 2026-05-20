# 🏠 Sri Lanka House Price Predictor

An interactive machine-learning dashboard for estimating Sri Lankan house prices from property details such as district, town, land size, house size, beds, baths, and seller type.

The project includes a trained model, a Streamlit valuation app, market-comparison charts, model-quality diagnostics, and scripts for retraining, evaluation, and explainability.

---

## ✨ Features

- 🧮 **Instant price prediction** from user-selected property inputs.
- 📍 **District and town-aware inputs** based on the cleaned dataset.
- 💰 **LKR formatted estimate** with lower and upper guide values.
- 📊 **Comparable market summary** for selected town or district.
- 📈 **Price distribution chart** comparing the prediction against local listings.
- 🧠 **Model explanation tab** with CatBoost SHAP-style local contribution drivers.
- ✅ **Model quality tab** with R2, MAE, RMSE, feature importance, and actual-vs-predicted plot.
- 🎨 **Custom Streamlit UI** with redesigned sidebar, tabs, cards, and responsive layout.

---

## 🧰 Tech Stack

- 🐍 Python
- 🌐 Streamlit
- 🐼 pandas
- 🔢 NumPy
- 📊 Matplotlib / Seaborn
- 🐱 CatBoost
- 🤖 scikit-learn
- 💾 joblib
- 🔍 SHAP

---

## 📁 Project Structure

```text
.
├── app/
│   ├── streamlit_app.py          # Main Streamlit dashboard
│   └── streamlit_app_old.py      # Older app version kept for reference
├── artifacts/
│   ├── catboost_house_price.cbm  # Trained CatBoost model
│   ├── model_meta.json           # Feature metadata, metrics, categories
│   ├── test_pred.npy             # Saved test predictions
│   └── test_true.npy             # Saved test target values
├── dataset/
│   └── cleaned_data_house_prices.csv
├── src/
│   ├── data.py                   # Dataset loading and cleaning helpers
│   ├── train.py                  # CatBoost training pipeline
│   ├── train_sklearn.py          # scikit-learn fallback training pipeline
│   ├── evaluate.py               # Evaluation metrics and plots
│   ├── explain.py                # Explainability reports
│   ├── metrics.py                # Regression metrics
│   └── plots.py                  # Saved evaluation plots
├── debug_fit.py                  # Small debug training helper
├── debug_fit_eval.py             # Small debug evaluation helper
├── Procfile                      # Deployment command for Streamlit hosting
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone or open the project

```bash
cd Sri-Lanka-House-Price-Predictor-main
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## 🖥️ How To Use The App

1. 📍 Choose a **District**.
2. 🏙️ Choose a **Town** filtered from available dataset towns.
3. 📐 Enter **Land size** in perches.
4. 🏡 Enter **House size** in square feet.
5. 🛏️ Select **Beds**.
6. 🛁 Select **Baths**.
7. 👤 Select **Seller type**.
8. ⚡ The prediction updates automatically.

The app shows:

- **Estimated market value**
- **Expected lower and upper range**
- **Current property summary**
- **Comparable listing statistics**
- **Price distribution chart**
- **Model explanations**
- **Model quality diagnostics**

---

## 🧠 Current Model

The included artifact uses a CatBoost regression model.

From `artifacts/model_meta.json`:

| Metric | Value |
|---|---:|
| Test R2 | `0.721` |
| Test MAE | about `LKR 7.6M` |
| Test RMSE | about `LKR 20.4M` |
| Train rows | `10,687` |
| Validation rows | `2,291` |
| Test rows | `2,291` |
| Boosting iterations | `700` |

> ⚠️ Price estimates are model-based guides, not official valuations. Real property prices can depend on road access, age, build quality, title status, neighborhood demand, and many details not captured in the dataset.

---

## 🏗️ Main App Functions

The main dashboard lives in `app/streamlit_app.py`.

| Function | Purpose |
|---|---|
| `format_lkr()` | Formats numeric values as readable LKR strings, such as `LKR 13.4M`. |
| `format_metric()` | Formats simple metric numbers for display. |
| `load_model_and_meta()` | Loads the trained model and metadata from `artifacts/`. |
| `load_market_data()` | Loads and cleans market data for comparable listings. |
| `district_town_map()` | Builds district-to-town dropdown options from the dataset. |
| `inputs_to_dataframe()` | Converts sidebar inputs into a one-row DataFrame for prediction. |
| `predict_price()` | Runs model inference and creates the price range. |
| `local_contributions()` | Gets local CatBoost feature contributions for the selected property. |
| `market_summary()` | Chooses town, district, or full-market comparable data depending on sample size. |
| `style_page()` | Injects custom CSS for the dashboard UI. |
| `render_header()` | Renders the top title, subtitle, and model badge. |
| `sidebar_inputs()` | Builds the sidebar controls for property inputs. |
| `render_price_panel()` | Displays the main estimated price card. |
| `render_property_summary()` | Displays selected property details. |
| `render_market_cards()` | Displays comparable area metrics. |
| `price_distribution_chart()` | Draws the local price distribution chart. |
| `contribution_chart()` | Draws local feature contribution bars. |
| `global_importance_chart()` | Draws global feature importance bars. |
| `actual_vs_predicted_chart()` | Draws actual-vs-predicted model quality plot. |
| `render_quality()` | Displays R2, MAE, and RMSE metrics. |
| `main()` | Runs the complete Streamlit app. |

---

## 🧹 Data Pipeline

Dataset utilities are in `src/data.py`.

| Function / Class | Purpose |
|---|---|
| `DatasetSchema` | Stores the target column and columns to drop. |
| `DEFAULT_SCHEMA` | Default schema using `Price` as the target. |
| `parse_first_float()` | Extracts the first valid number from messy text fields. |
| `load_dataset()` | Loads the tab-separated CSV, cleans columns, converts numeric fields, and drops missing targets. |
| `infer_feature_types()` | Splits columns into feature, categorical, and numeric lists. |
| `top_categories()` | Collects the most common categorical values for UI dropdowns. |

Important note:

```text
dataset/cleaned_data_house_prices.csv
```

has a `.csv` extension, but the loader reads it as **tab-separated** data.

---

## 🏋️ Retrain The CatBoost Model

Run:

```bash
python -m src.train --dataset dataset/cleaned_data_house_prices.csv --artifacts artifacts --trials 3
```

Useful options:

```bash
python -m src.train --seed 42 --trials 3
```

The training script:

1. 📥 Loads the cleaned dataset.
2. 🧼 Infers numeric and categorical features.
3. ✂️ Splits data into train, validation, and test sets.
4. 🧪 Tests a small CatBoost parameter grid.
5. 🏆 Selects the best validation model by R2.
6. 💾 Saves the final model and metadata into `artifacts/`.

Generated files:

```text
artifacts/catboost_house_price.cbm
artifacts/model_meta.json
artifacts/test_true.npy
artifacts/test_pred.npy
```

---

## 🔁 Train The scikit-learn Fallback Model

If CatBoost is unavailable or you want a pure scikit-learn pipeline:

```bash
python -m src.train_sklearn --dataset dataset/cleaned_data_house_prices.csv --artifacts artifacts
```

This creates:

```text
artifacts/hgb_house_price.joblib
artifacts/model_meta.json
artifacts/test_true.npy
artifacts/test_pred.npy
```

The Streamlit app can load this fallback artifact if present.

---

## 📊 Evaluate The Model

Run:

```bash
python -m src.evaluate --artifacts artifacts --figures reports/figures
```

This saves:

```text
reports/figures/test_metrics.json
reports/figures/test_pred_vs_actual.png
reports/figures/test_residuals_hist.png
reports/figures/test_residuals_vs_pred.png
```

---

## 🔍 Generate Explainability Reports

For CatBoost explainability:

```bash
python -m src.explain --dataset dataset/cleaned_data_house_prices.csv --artifacts artifacts --figures reports/figures
```

This can create:

```text
reports/figures/feature_importance.png
reports/figures/shap_summary.png
reports/figures/pdp.png
```

---

## 🌐 Deployment

The project includes a `Procfile`:

```text
web: streamlit run app/streamlit_app.py --server.port $PORT
```

This is useful for platforms that provide a `$PORT` environment variable.

Before deploying, make sure these files are included:

- `app/streamlit_app.py`
- `src/`
- `requirements.txt`
- `artifacts/model_meta.json`
- either `artifacts/catboost_house_price.cbm` or `artifacts/hgb_house_price.joblib`
- `dataset/cleaned_data_house_prices.csv`

---

## 🧪 Troubleshooting

### ❌ `No usable model artifact was found`

Make sure one of these exists:

```text
artifacts/catboost_house_price.cbm
artifacts/hgb_house_price.joblib
```

If missing, retrain the model:

```bash
python -m src.train --dataset dataset/cleaned_data_house_prices.csv --artifacts artifacts --trials 3
```

### ❌ CatBoost import error

Install requirements again:

```bash
pip install -r requirements.txt
```

Or use the scikit-learn training path:

```bash
python -m src.train_sklearn
```

### ❌ Streamlit app does not open

Try:

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Then open:

```text
http://localhost:8501
```

### ❌ Dataset loading looks wrong

The dataset file is read using tab separation:

```python
pd.read_csv(path, sep="\t", engine="python")
```

Do not manually convert it to comma-separated format unless you also update `src/data.py`.

---

## 💡 Suggested Improvements

- 🗺️ Add map-based neighborhood visualization.
- 🏘️ Add property type, road width, age, and furnishing fields if data becomes available.
- 📦 Add Docker support for easier deployment.
- 🧪 Add unit tests for data loading and prediction helpers.
- 📉 Add confidence interval calibration instead of using MAE/spread heuristics.
- 🔐 Add validation for unusual user inputs.

---

## 📌 Notes

- The current UI is optimized for desktop and wide screens.
- The prediction range is a guide based on model error and predicted value.
- Model quality depends heavily on dataset coverage and cleanliness.
- Some locations may fall back from town-level comparable data to district-level or all-area data if there are too few listings.

---

## 🙌 Credits

Built as a Sri Lankan real-estate price prediction project using cleaned house listing data, CatBoost regression, and a custom Streamlit dashboard.

