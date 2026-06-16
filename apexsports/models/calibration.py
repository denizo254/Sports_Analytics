"""Calibration utilities — compare our xG model against StatsBomb's xG.

A reliability diagram bins shots by predicted probability and plots the mean
predicted xG against the observed goal frequency in each bin; a perfectly
calibrated model lies on the diagonal. We compute this for both our logistic
xG model and StatsBomb's reference xG (when real data is loaded), alongside
scalar scores (Brier, log loss, AUC) so the two can be ranked against the
actual outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from apexsports.data.database import get_session
from apexsports.data.schema import Shot
from apexsports.models import xg as xg_model


def load_shot_predictions(model=None) -> pd.DataFrame:
    """One row per shot with our predicted xG, StatsBomb xG, and the outcome."""
    model = model or xg_model.load_model()
    with get_session() as s:
        rows = s.query(
            Shot.distance, Shot.angle, Shot.is_header, Shot.under_pressure,
            Shot.big_chance, Shot.sb_xg, Shot.is_goal,
        ).all()
    df = pd.DataFrame(rows, columns=[
        "distance", "angle", "is_header", "under_pressure", "big_chance",
        "sb_xg", "is_goal"])
    if df.empty:
        return df
    for col in ("is_header", "under_pressure", "big_chance", "is_goal"):
        df[col] = df[col].astype(int)
    df["our_xg"] = model.predict_proba(df[xg_model.FEATURES])[:, 1]
    return df


def reliability_curve(probs, outcomes, n_bins: int = 10) -> pd.DataFrame:
    """Bin by predicted prob; return mean predicted vs observed frequency."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)

    out = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        out.append({
            "bin": b,
            "mean_predicted": float(probs[mask].mean()),
            "observed_freq": float(outcomes[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(out)


def score(probs, outcomes) -> dict:
    """Scalar calibration / discrimination metrics against actual goals."""
    probs = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    outcomes = np.asarray(outcomes, dtype=int)
    metrics = {
        "brier": float(brier_score_loss(outcomes, probs)),
        "log_loss": float(log_loss(outcomes, probs, labels=[0, 1])),
        "mean_xg": float(probs.mean()),
        "actual_rate": float(outcomes.mean()),
    }
    # AUC needs both classes present.
    if len(np.unique(outcomes)) == 2:
        metrics["auc"] = float(roc_auc_score(outcomes, probs))
    return metrics


def compare(n_bins: int = 10, model=None) -> dict:
    """Full comparison payload for the dashboard / API.

    Returns reliability curves + scores for our model, plus StatsBomb's when
    reference xG is present (real data). `has_statsbomb` is False on synthetic
    data, where sb_xg is all zeros.
    """
    df = load_shot_predictions(model=model)
    if df.empty:
        raise RuntimeError("No shots in DB — load or generate data first.")

    has_sb = bool(df["sb_xg"].sum() > 0)
    payload = {
        "n_shots": int(len(df)),
        "has_statsbomb": has_sb,
        "our": {
            "curve": reliability_curve(df["our_xg"], df["is_goal"], n_bins)
            .to_dict("records"),
            "score": score(df["our_xg"], df["is_goal"]),
        },
    }
    if has_sb:
        payload["statsbomb"] = {
            "curve": reliability_curve(df["sb_xg"], df["is_goal"], n_bins)
            .to_dict("records"),
            "score": score(df["sb_xg"], df["is_goal"]),
        }
        # Agreement between the two models, shot for shot.
        payload["agreement"] = {
            "pearson_r": float(np.corrcoef(df["our_xg"], df["sb_xg"])[0, 1]),
            "mean_abs_diff": float((df["our_xg"] - df["sb_xg"]).abs().mean()),
        }
    return payload


if __name__ == "__main__":
    import json
    print(json.dumps(compare(), indent=2, default=str))
