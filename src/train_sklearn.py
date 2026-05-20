from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import TransformedTargetRegressor

from src.data import DEFAULT_SCHEMA, infer_feature_types, load_dataset, top_categories
from src.metrics import regression_metrics


@dataclass
class TrainConfig:
    dataset_path: str
    artifacts_dir: str
    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/cleaned_data_house_prices.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(dataset_path=args.dataset, artifacts_dir=args.artifacts, seed=args.seed)
    artifacts_dir = Path(cfg.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("[train_sklearn] loading dataset...", flush=True)
    df = load_dataset(cfg.dataset_path, schema=DEFAULT_SCHEMA)
    target = DEFAULT_SCHEMA.target

    feature_columns, categorical_columns, numeric_columns = infer_feature_types(df, target=target)

    train_val, test = train_test_split(df, test_size=cfg.test_size, random_state=cfg.seed)
    val_frac = cfg.val_size / (1.0 - cfg.test_size)
    train, val = train_test_split(train_val, test_size=val_frac, random_state=cfg.seed)

    def split_xy(d):
        return d[feature_columns].copy(), d[target].to_numpy(dtype=float)

    x_train, y_train = split_xy(train)
    x_val, y_val = split_xy(val)
    x_test, y_test = split_xy(test)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True),
            ),
        ]
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_columns),
            ("cat", categorical_pipe, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    base = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=8,
        max_iter=600,
        l2_regularization=0.0,
        random_state=cfg.seed,
    )

    # Train in log-price space for more stable errors (house prices are heavy-tailed).
    model = Pipeline(
        steps=[
            ("pre", pre),
            (
                "reg",
                TransformedTargetRegressor(
                    regressor=base,
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ]
    )

    print("[train_sklearn] fitting model...", flush=True)
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    val_metrics = regression_metrics(y_val, val_pred)
    test_metrics = regression_metrics(y_test, test_pred)

    print("[train_sklearn] val metrics:", val_metrics, flush=True)
    print("[train_sklearn] test metrics:", test_metrics, flush=True)

    joblib.dump(model, artifacts_dir / "hgb_house_price.joblib")

    categories = top_categories(df, categorical_columns, max_values=200)

    meta = {
        "model_type": "sklearn_hgb_onehot",
        "target": target,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "splits": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "categories": categories,
        "notes": {
            "target_transform": "log1p",
            "onehot_min_frequency": 10,
        },
    }

    (artifacts_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.save(artifacts_dir / "test_true.npy", y_test)
    np.save(artifacts_dir / "test_pred.npy", test_pred)

    print("[train_sklearn] saved artifacts to", artifacts_dir, flush=True)


if __name__ == "__main__":
    main()
