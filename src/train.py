from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

from src.data import DEFAULT_SCHEMA, infer_feature_types, load_dataset, top_categories
from src.metrics import regression_metrics


TargetTransform = Literal["none", "log1p"]


@dataclass
class TrainConfig:
    dataset_path: str
    artifacts_dir: str
    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    max_cat_values: int = 200
    n_trials: int = 25


def _split_train_val_test(
    df: pd.DataFrame,
    *,
    target: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed)
    # val_size is relative to full dataset; convert to fraction of train_val
    val_frac = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=val_frac, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _prepare_pools(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target: str,
    feature_columns: List[str],
    categorical_columns: List[str],
    transform: TargetTransform,
) -> tuple[Pool, Pool, Pool, np.ndarray, np.ndarray, np.ndarray]:
    def _y(df: pd.DataFrame) -> np.ndarray:
        y = df[target].to_numpy(dtype=float)
        if transform == "log1p":
            # target is price, strictly positive in this dataset.
            y = np.log1p(np.maximum(y, 0.0))
        return y

    def _x(df: pd.DataFrame) -> pd.DataFrame:
        x = df[feature_columns].copy()
        for c in categorical_columns:
            x[c] = x[c].astype(str).fillna("__MISSING__")
        return x

    train_x, val_x, test_x = _x(train_df), _x(val_df), _x(test_df)
    train_y, val_y, test_y = _y(train_df), _y(val_df), _y(test_df)

    cat_idx = [feature_columns.index(c) for c in categorical_columns]

    train_pool = Pool(train_x, label=train_y, cat_features=cat_idx)
    val_pool = Pool(val_x, label=val_y, cat_features=cat_idx)
    test_pool = Pool(test_x, label=test_y, cat_features=cat_idx)
    return train_pool, val_pool, test_pool, train_y, val_y, test_y


def _inverse_transform(pred: np.ndarray, transform: TargetTransform) -> np.ndarray:
    if transform == "log1p":
        return np.expm1(pred)
    return pred


def _fit_one(
    *,
    train_pool: Pool,
    val_pool: Pool,
    params: Dict[str, Any],
    seed: int,
) -> CatBoostRegressor:
    params = dict(params)
    iterations = int(params.pop("iterations", 2000))
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        random_seed=seed,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
        **params,
    )
    # Fit on train only (tuning evaluates on val separately). This avoids
    # issues with long-running eval_set + early-stopping in some environments.
    model.fit(train_pool)
    return model


def train_and_select(
    df: pd.DataFrame,
    *,
    target: str,
    feature_columns: List[str],
    categorical_columns: List[str],
    seed: int,
    test_size: float,
    val_size: float,
    n_trials: int,
) -> dict:
    train_df, val_df, test_df = _split_train_val_test(
        df, target=target, seed=seed, test_size=test_size, val_size=val_size
    )

    best: dict | None = None

    # A small, stable hyperparameter grid (no exotic sampling/bootstraps).
    # Keep it simple and reproducible for coursework.
    param_grid: list[dict[str, Any]] = [
        {"iterations": 600, "depth": 6, "learning_rate": 0.10, "l2_leaf_reg": 3.0, "random_strength": 1.0, "min_data_in_leaf": 10},
        {"iterations": 700, "depth": 8, "learning_rate": 0.08, "l2_leaf_reg": 5.0, "random_strength": 1.0, "min_data_in_leaf": 20},
        {"iterations": 900, "depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 8.0, "random_strength": 1.0, "min_data_in_leaf": 20},
    ]

    for transform in ("none", "log1p"):
        train_pool, val_pool, test_pool, _, _, _ = _prepare_pools(
            train_df,
            val_df,
            test_df,
            target=target,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            transform=transform,
        )

        for trial in range(min(n_trials, len(param_grid))):
            params = param_grid[trial]
            try:
                iters = int(params.get("iterations", 0))
                print(
                    f"trial {trial+1:02d}/{n_trials} | transform={transform} | fitting (iters={iters})...",
                    flush=True,
                )
                model = _fit_one(train_pool=train_pool, val_pool=val_pool, params=params, seed=seed)
                print(
                    f"trial {trial+1:02d}/{n_trials} | transform={transform} | fit done (trees={model.tree_count_})",
                    flush=True,
                )
            except Exception as exc:
                print(f"trial {trial+1:02d}/{n_trials} | transform={transform} | FAILED: {exc}", flush=True)
                continue

            val_pred = _inverse_transform(model.predict(val_pool), transform)
            val_true = val_df[target].to_numpy(dtype=float)
            metrics = regression_metrics(val_true, val_pred)

            record = {
                "transform": transform,
                "params": params,
                "best_iteration": int(model.tree_count_),
                "val_metrics": metrics,
            }

            if best is None or metrics["r2"] > best["val_metrics"]["r2"]:
                best = record

            print(
                f"trial {trial+1:02d}/{n_trials} | transform={transform} | "
                f"val_r2={metrics['r2']:.4f} | val_rmse={metrics['rmse']:.0f}"
            , flush=True)

    assert best is not None

    # Refit on train+val using chosen transform/params and chosen iteration count.
    transform: TargetTransform = best["transform"]
    params = best["params"]
    final_iterations = int(params.get("iterations", best["best_iteration"]))
    final_params = dict(params)
    final_params.pop("iterations", None)

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    train_pool, val_pool, test_pool, _, _, _ = _prepare_pools(
        train_val_df,
        val_df,  # dummy, not used for refit
        test_df,
        target=target,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        transform=transform,
    )

    # Fit using fixed number of trees from the selected run (best_iteration).
    final_model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=final_iterations,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        **final_params,
    )
    final_model.fit(train_pool)

    test_pred = _inverse_transform(final_model.predict(test_pool), transform)
    test_true = test_df[target].to_numpy(dtype=float)
    test_metrics = regression_metrics(test_true, test_pred)

    return {
        "final_model": final_model,
        "transform": transform,
        "best_params": params,
        "best_iteration": int(best["best_iteration"]),
        "val_metrics": best["val_metrics"],
        "test_metrics": test_metrics,
        "splits": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "test_pred": test_pred,
        "test_true": test_true,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/cleaned_data_house_prices.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=25)
    args = parser.parse_args()

    print("[train] starting", flush=True)

    cfg = TrainConfig(
        dataset_path=args.dataset,
        artifacts_dir=args.artifacts,
        seed=args.seed,
        n_trials=args.trials,
    )

    artifacts_dir = Path(cfg.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_dataset(cfg.dataset_path, schema=DEFAULT_SCHEMA)
    except Exception as exc:
        print(f"[train] failed to load dataset: {exc}", flush=True)
        raise

    print(f"[train] loaded dataset: shape={df.shape}", flush=True)

    target = DEFAULT_SCHEMA.target
    feature_columns, categorical_columns, numeric_columns = infer_feature_types(df, target=target)

    # Keep a small amount of metadata for the UI.
    categories = top_categories(df, categorical_columns, max_values=cfg.max_cat_values)

    print("[train] tuning model...", flush=True)
    result = train_and_select(
        df,
        target=target,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        seed=cfg.seed,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        n_trials=cfg.n_trials,
    )

    model: CatBoostRegressor = result["final_model"]

    model_path = artifacts_dir / "catboost_house_price.cbm"
    model.save_model(model_path)

    meta = {
        "target": target,
        "feature_columns": result["feature_columns"],
        "categorical_columns": result["categorical_columns"],
        "numeric_columns": numeric_columns,
        "target_transform": result["transform"],
        "best_params": result["best_params"],
        "best_iteration": result["best_iteration"],
        "splits": result["splits"],
        "val_metrics": result["val_metrics"],
        "test_metrics": result["test_metrics"],
        "categories": categories,
    }

    (artifacts_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Save test preds for plots/explainability.
    np.save(artifacts_dir / "test_true.npy", result["test_true"])
    np.save(artifacts_dir / "test_pred.npy", result["test_pred"])

    print("\nSaved model to:", model_path)
    print("Saved metadata to:", artifacts_dir / "model_meta.json")
    print("\nTest metrics:")
    for k, v in result["test_metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}" if k == "r2" else f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
