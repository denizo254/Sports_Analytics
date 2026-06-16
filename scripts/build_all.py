"""End-to-end pipeline: load/generate data -> train all models -> save artifacts.

Synthetic (default):
    python scripts/build_all.py

Real StatsBomb open data (needs `pip install statsbombpy`):
    python scripts/build_all.py --source statsbomb            # 2022 FIFA World Cup
    python scripts/build_all.py --source statsbomb --competition 43 --season 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/build_all.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apexsports.data import generate
from apexsports.models import xg, poisson, forecast


def main() -> None:
    ap = argparse.ArgumentParser(description="ApexSports build pipeline")
    ap.add_argument("--source", choices=["synthetic", "statsbomb", "fbref"],
                    default="synthetic")
    ap.add_argument("--competition", default=None,
                    help="statsbomb: competition_id (default 43 = FIFA World "
                         "Cup). fbref: league name (default Champions League)")
    ap.add_argument("--season", nargs="+", default=None,
                    help="statsbomb: season_ids (e.g. 3 106 = 2018+2022 WC). "
                         "fbref: season strings (e.g. 2024-2025 2025-2026)")
    args = ap.parse_args()

    print("=" * 60)
    print(f"ApexSports Analytics — build pipeline ({args.source})")
    print("=" * 60)

    print(f"\n[1/5] Loading data ({args.source})...")
    if args.source == "statsbomb":
        from apexsports.data.statsbomb import load_competitions
        comp = int(args.competition or 43)
        seasons = args.season or [106]
        specs = [(comp, int(s)) for s in seasons]
        counts = load_competitions(specs, verbose=False)
    elif args.source == "fbref":
        from apexsports.data.fbref import load_fbref, UCL_LEAGUE
        seasons = args.season or ["2024-2025", "2025-2026"]
        counts = load_fbref(seasons, competition=args.competition or UCL_LEAGUE,
                            verbose=False)
    else:
        counts = generate.generate()
    print(json.dumps(counts, indent=2))

    print("\n[2/5] Training xG model (logistic regression)...")
    try:
        xg_metrics = xg.train()
        print(f"  AUC={xg_metrics['auc']:.3f}  logloss={xg_metrics['log_loss']:.3f}"
              f"  Brier={xg_metrics['brier']:.3f}")
        print(f"  coefficients: {xg_metrics['coefficients']}")
    except RuntimeError as e:
        print(f"  skipped (no shot-level data for this source): {e}")

    print("\n[3/5] Building Poisson player-goal ratings...")
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
    except RuntimeError as e:
        print(f"  skipped: {e}")

    print("\nDone. Artifacts written to ./artifacts/")
    print("Next: `uvicorn apexsports.api.main:app --reload`  or  "
          "`streamlit run apexsports/dashboard/app.py`")


if __name__ == "__main__":
    main()
