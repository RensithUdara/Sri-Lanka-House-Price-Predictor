from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.metrics import regression_metrics
from src.plots import save_evaluation_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--figures", default="reports/figures")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)
    figures_dir = Path(args.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.load(artifacts_dir / "test_true.npy")
    y_pred = np.load(artifacts_dir / "test_pred.npy")

    metrics = regression_metrics(y_true, y_pred)

    (figures_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_evaluation_plots(y_true, y_pred, out_dir=figures_dir, prefix="test")

    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if k == "r2" else f"  {k}: {v:.2f}")
    print("\nSaved plots to:", figures_dir)


if __name__ == "__main__":
    main()
