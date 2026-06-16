"""End-to-end pipeline: generate data -> train all models -> save artifacts.

Run once before launching the API or dashboard:

    python scripts/build_all.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as `python scripts/build_all.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apexsports.data import generate
from apexsports.models import xg, poisson, forecast


def main() -> None:
    print("=" * 60)
    print("ApexSports Analytics — build pipeline")
    print("=" * 60)

    print("\n[1/4] Generating synthetic tournament data...")
    counts = generate.generate()
    print(json.dumps(counts, indent=2))

    print("\n[2/4] Training xG model (logistic regression)...")
    xg_metrics = xg.train()
    print(f"  AUC={xg_metrics['auc']:.3f}  logloss={xg_metrics['log_loss']:.3f}  "
          f"Brier={xg_metrics['brier']:.3f}")
    print(f"  coefficients: {xg_metrics['coefficients']}")

    print("\n[3/4] Building Poisson player-goal ratings...")
    pois = poisson.build_ratings()
    print(f"  rated {len(pois['players'])} players, "
          f"{len(pois['defence'])} team defences")

    print("\n[4/5] Training XGBoost performance forecaster...")
    fc = forecast.train()
    print(f"  rows={fc['n_rows']}  MAE={fc['mae']:.4f}  R2={fc['r2']:.3f}")
    print(f"  top features: {list(fc['feature_importance'].items())[:3]}")

    print("\n[5/5] Training LSTM sequence forecaster (PyTorch)...")
    try:
        from apexsports.models import lstm_forecast
        lf = lstm_forecast.train()
        print(f"  samples={lf['n_samples']}  window={lf['window']}  "
              f"MAE={lf['mae']:.4f}  R2={lf['r2']:.3f}  "
              f"(best epoch {lf['best_epoch']})")
    except ImportError:
        print("  torch not installed — skipping LSTM. "
              "`pip install torch` to enable.")

    print("\nDone. Artifacts written to ./artifacts/")
    print("Next: `uvicorn apexsports.api.main:app --reload`  or  "
          "`streamlit run apexsports/dashboard/app.py`")


if __name__ == "__main__":
    main()
