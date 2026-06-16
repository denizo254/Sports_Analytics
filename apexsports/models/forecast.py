"""Player performance forecasting (XGBoost).

Predicts a player's expected attacking output (xG) for an upcoming fixture from
recent form plus tournament-context features (fatigue index, rest days, travel
distance, stadium elevation). This is the regression baseline called for in
Phase 2 of the delivery plan; an LSTM variant can later consume the same
feature frame.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from config import FORECAST_MODEL_PATH, RANDOM_SEED
from apexsports.data.database import get_session
from apexsports.data.schema import Match, Player, PlayerMatchStat

POS_CODES = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
FEATURES = [
    "skill", "position_code", "rest_days", "travel_km", "elevation_m",
    "fatigue_index", "form_xg3", "form_minutes3", "career_xg90",
]
TARGET = "xg"


def _build_frame() -> pd.DataFrame:
    with get_session() as s:
        rows = s.query(
            PlayerMatchStat.player_id, PlayerMatchStat.match_id,
            PlayerMatchStat.minutes, PlayerMatchStat.xg, PlayerMatchStat.goals,
            PlayerMatchStat.rest_days, PlayerMatchStat.travel_km,
            PlayerMatchStat.elevation_m, PlayerMatchStat.fatigue_index,
            Match.date, Player.position, Player.skill,
        ).join(Match, Match.id == PlayerMatchStat.match_id) \
         .join(Player, Player.id == PlayerMatchStat.player_id).all()

    df = pd.DataFrame(rows, columns=[
        "player_id", "match_id", "minutes", "xg", "goals", "rest_days",
        "travel_km", "elevation_m", "fatigue_index", "date", "position", "skill"])
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    df["position_code"] = df["position"].map(POS_CODES)

    g = df.groupby("player_id", group_keys=False)
    # Rolling form, shifted so only PAST matches inform each row (no leakage).
    df["form_xg3"] = g["xg"].apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["form_minutes3"] = g["minutes"].apply(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    # Career xG/90 from past matches only (series-level groupby; no leakage).
    cum_xg = g["xg"].apply(lambda x: x.shift(1).cumsum())
    cum_min = g["minutes"].apply(lambda x: x.shift(1).cumsum())
    df["career_xg90"] = cum_xg / (cum_min / 90).replace(0, np.nan)

    df[["form_xg3", "form_minutes3", "career_xg90"]] = \
        df[["form_xg3", "form_minutes3", "career_xg90"]].fillna(0.0)
    # Drop the first appearance per player (no history to learn from).
    df = df[df.groupby("player_id").cumcount() > 0]
    return df


def train(save: bool = True) -> dict:
    df = _build_frame()
    if df.empty:
        raise RuntimeError("Not enough match history to train forecaster.")

    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED)

    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED,
        objective="reg:squarederror")
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    importances = dict(sorted(
        zip(FEATURES, model.feature_importances_.round(4).tolist()),
        key=lambda kv: kv[1], reverse=True))
    metrics = {
        "n_rows": int(len(df)),
        "mae": float(mean_absolute_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
        "feature_importance": importances,
    }
    if save:
        joblib.dump({"model": model, "features": FEATURES}, FORECAST_MODEL_PATH)
        metrics["saved_to"] = str(FORECAST_MODEL_PATH)
    return metrics


def load_model():
    if not FORECAST_MODEL_PATH.exists():
        raise FileNotFoundError("Forecast model missing. Run scripts/build_all.py")
    return joblib.load(FORECAST_MODEL_PATH)


def forecast_player(features: dict, bundle=None) -> dict:
    """Predict expected xG for an upcoming fixture from a feature dict."""
    bundle = bundle or load_model()
    model, feat_names = bundle["model"], bundle["features"]
    row = np.array([[float(features.get(f, 0.0)) for f in feat_names]])
    pred = float(model.predict(row)[0])
    return {"predicted_xg": round(max(0.0, pred), 4), "inputs": features}


if __name__ == "__main__":
    import json
    print(json.dumps(train(), indent=2))
