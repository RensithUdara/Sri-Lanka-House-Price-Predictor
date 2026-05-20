from __future__ import annotations

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

from src.data import DEFAULT_SCHEMA, infer_feature_types, load_dataset


def main() -> None:
    df = load_dataset("dataset/cleaned_data_house_prices.csv", schema=DEFAULT_SCHEMA)
    target = "Price"
    feats, cat_cols, _ = infer_feature_types(df, target=target)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    def x(d):
        xx = d[feats].copy()
        for c in cat_cols:
            xx[c] = xx[c].astype(str).fillna("__MISSING__")
        return xx

    y_train = train_df[target].to_numpy(float)
    y_val = val_df[target].to_numpy(float)

    cat_idx = [feats.index(c) for c in cat_cols]

    train_pool = Pool(x(train_df), label=y_train, cat_features=cat_idx)
    val_pool = Pool(x(val_df), label=y_val, cat_features=cat_idx)

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2000,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        random_seed=42,
        verbose=200,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=200,
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    print("done", model.tree_count_)


if __name__ == "__main__":
    main()
