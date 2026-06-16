"""Expected Goals (xG) model.

Implements the standard xG framework as logistic regression on shot geometry:

    log(p / (1 - p)) = b0 + b1*distance + b2*angle + b3*is_header + ...

The fitted coefficients should approximate the synthetic ground truth in
generate.TRUE_BETA, which is asserted in the test suite.
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from config import XG_MODEL_PATH, RANDOM_SEED
from apexsports.data.database import get_session
from apexsports.data.schema import Shot
from apexsports.utils import shot_geometry

FEATURES = ["distance", "angle", "is_header", "under_pressure", "big_chance"]


def load_shots() -> pd.DataFrame:
    with get_session() as s:
        rows = s.query(
            Shot.distance, Shot.angle, Shot.is_header, Shot.under_pressure,
            Shot.big_chance, Shot.is_goal,
        ).all()
    df = pd.DataFrame(rows, columns=FEATURES + ["is_goal"])
    for col in ["is_header", "under_pressure", "big_chance", "is_goal"]:
        df[col] = df[col].astype(int)
    return df


def train(save: bool = True) -> dict:
    df = load_shots()
    if df.empty:
        raise RuntimeError("No shots in DB — run the data generator first.")

    X, y = df[FEATURES], df["is_goal"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y)

    model = LogisticRegression(max_iter=1000, C=5.0)
    model.fit(X_tr, y_tr)

    p_te = model.predict_proba(X_te)[:, 1]
    metrics = {
        "n_shots": int(len(df)),
        "conversion_rate": float(y.mean()),
        "auc": float(roc_auc_score(y_te, p_te)),
        "log_loss": float(log_loss(y_te, p_te)),
        "brier": float(brier_score_loss(y_te, p_te)),
        "coefficients": dict(zip(FEATURES, model.coef_[0].round(4).tolist())),
        "intercept": float(round(model.intercept_[0], 4)),
    }
    if save:
        joblib.dump(model, XG_MODEL_PATH)
        metrics["saved_to"] = str(XG_MODEL_PATH)
    return metrics


def load_model() -> LogisticRegression:
    if not XG_MODEL_PATH.exists():
        raise FileNotFoundError("xG model not trained. Run scripts/build_all.py")
    return joblib.load(XG_MODEL_PATH)


def predict_xg(x: float, y: float, is_header: bool = False,
               under_pressure: bool = False, big_chance: bool = False,
               model: LogisticRegression | None = None) -> dict:
    """Compute xG for a single shot from raw pitch coordinates."""
    model = model or load_model()
    distance, angle = shot_geometry(x, y)
    feats = pd.DataFrame(
        [[distance, angle, int(is_header), int(under_pressure), int(big_chance)]],
        columns=FEATURES)
    p = float(model.predict_proba(feats)[0, 1])
    return {"xg": round(p, 4), "distance": round(distance, 2),
            "angle_rad": round(angle, 4), "x": x, "y": y}


if __name__ == "__main__":
    import json
    print(json.dumps(train(), indent=2))
