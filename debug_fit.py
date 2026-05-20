from __future__ import annotations

from catboost import CatBoostRegressor, Pool

from src.data import DEFAULT_SCHEMA, infer_feature_types, load_dataset


def main() -> None:
    df = load_dataset("dataset/cleaned_data_house_prices.csv", schema=DEFAULT_SCHEMA)
    print("loaded", df.shape)

    target = "Price"
    features, cat_cols, _ = infer_feature_types(df, target=target)

    x = df[features].copy()
    for c in cat_cols:
        x[c] = x[c].astype(str).fillna("__MISSING__")

    y = df[target].to_numpy(dtype=float)
    cat_idx = [features.index(c) for c in cat_cols]

    pool = Pool(x, label=y, cat_features=cat_idx)

    model = CatBoostRegressor(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        loss_function="RMSE",
        verbose=50,
        allow_writing_files=False,
        random_seed=42,
    )

    model.fit(pool)
    print("done", model.tree_count_)


if __name__ == "__main__":
    main()
