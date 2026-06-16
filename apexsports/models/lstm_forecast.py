"""LSTM player-performance forecaster (PyTorch).

The sequence counterpart to the tabular XGBoost forecaster. It ingests a
player's *ordered sequence* of recent match feature vectors, encodes the
temporal pattern with an LSTM, then fuses the upcoming match's KNOWN pre-match
context (rest days, travel, elevation, expected minutes) before projecting
next-match xG. Including the upcoming context is both realistic — those values
are known before kickoff — and gives the model the same footing as the XGBoost
baseline, which also conditions on the target match's context.

Architecture:
    past matches --LSTM--> final hidden state ┐
                                              ├─ concat ─> MLP head -> xG
    upcoming match known context ─────────────┘

torch is an optional/heavy dependency, so this module is imported lazily by the
build pipeline and the API; its test is skipped when torch is absent.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from config import LSTM_MODEL_PATH, LSTM_SCALER_PATH, RANDOM_SEED
from apexsports.data.database import get_session
from apexsports.data.schema import Match, Player, PlayerMatchStat

POS_CODES = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}

# Per-timestep features describing each PAST match in the sequence.
SEQ_FEATURES = [
    "minutes", "xg", "goals", "shots", "passes", "distance_km", "assists",
    "rest_days", "travel_km", "elevation_m", "fatigue_index",
    "skill", "position_code",
]
# Known-before-kickoff context of the UPCOMING (target) match.
CTX_FEATURES = [
    "minutes", "rest_days", "travel_km", "elevation_m", "fatigue_index",
    "skill", "position_code",
]
WINDOW = 4          # matches of history per sample
TARGET = "xg"       # next-match xG


class PlayerLSTM(nn.Module):
    def __init__(self, seq_features: int, ctx_features: int,
                 hidden_size: int = 16, num_layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(seq_features, hidden_size, num_layers,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + ctx_features, 16), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(16, 1))

    def forward(self, seq, ctx):               # seq:(B,T,F_seq) ctx:(B,F_ctx)
        _, (h, _) = self.lstm(seq)
        z = torch.cat([h[-1], ctx], dim=1)
        return self.head(z).squeeze(-1)


def _player_frame() -> pd.DataFrame:
    """One ordered row per player-match with all needed columns."""
    with get_session() as s:
        rows = s.query(
            PlayerMatchStat.player_id, PlayerMatchStat.minutes,
            PlayerMatchStat.xg, PlayerMatchStat.goals, PlayerMatchStat.shots,
            PlayerMatchStat.passes, PlayerMatchStat.distance_km,
            PlayerMatchStat.assists, PlayerMatchStat.rest_days,
            PlayerMatchStat.travel_km, PlayerMatchStat.elevation_m,
            PlayerMatchStat.fatigue_index, Match.date,
            Player.skill, Player.position,
        ).join(Match, Match.id == PlayerMatchStat.match_id) \
         .join(Player, Player.id == PlayerMatchStat.player_id).all()

    df = pd.DataFrame(rows, columns=[
        "player_id", "minutes", "xg", "goals", "shots", "passes",
        "distance_km", "assists", "rest_days", "travel_km", "elevation_m",
        "fatigue_index", "date", "skill", "position"])
    df["position_code"] = df["position"].map(POS_CODES)
    return df.sort_values(["player_id", "date"]).reset_index(drop=True)


def _build_samples(df: pd.DataFrame):
    """Per player: window of past matches -> (seq, upcoming-ctx, target xG)."""
    seqs, ctxs, ys = [], [], []
    for _, grp in df.groupby("player_id"):
        seq_arr = grp[SEQ_FEATURES].to_numpy(dtype=np.float32)
        ctx_arr = grp[CTX_FEATURES].to_numpy(dtype=np.float32)
        tgt = grp[TARGET].to_numpy(dtype=np.float32)
        for i in range(len(grp) - WINDOW):
            seqs.append(seq_arr[i:i + WINDOW])
            ctxs.append(ctx_arr[i + WINDOW])      # the upcoming match's context
            ys.append(tgt[i + WINDOW])
    if not seqs:
        raise RuntimeError(
            f"Not enough match history per player for WINDOW={WINDOW}. "
            "Generate more rounds (data/generate.py).")
    return (np.stack(seqs), np.stack(ctxs), np.array(ys, dtype=np.float32))


def train(epochs: int = 300, hidden_size: int = 16, lr: float = 1e-3,
          batch_size: int = 64, patience: int = 25, save: bool = True,
          seed: int = RANDOM_SEED) -> dict:
    import copy

    torch.manual_seed(seed)
    np.random.seed(seed)

    df = _player_frame()
    seq, ctx, y = _build_samples(df)

    # Three-way split: fit / val (early stopping) / test (held-out report).
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(seq))
    n_te = int(len(seq) * 0.20)
    n_val = int(len(seq) * 0.15)
    te, val, fit = perm[:n_te], perm[n_te:n_te + n_val], perm[n_te + n_val:]

    # Scalers fit on the FIT split only (no leakage into val/test).
    n_seq_f, n_ctx_f = seq.shape[2], ctx.shape[1]
    seq_scaler = StandardScaler().fit(seq[fit].reshape(-1, n_seq_f))
    ctx_scaler = StandardScaler().fit(ctx[fit])

    def _ss(a):
        return torch.from_numpy(
            seq_scaler.transform(a.reshape(-1, n_seq_f))
            .reshape(a.shape).astype(np.float32))

    def _cs(a):
        return torch.from_numpy(ctx_scaler.transform(a).astype(np.float32))

    seq_fit, ctx_fit, y_fit = _ss(seq[fit]), _cs(ctx[fit]), torch.from_numpy(y[fit])
    seq_val, ctx_val, y_val = _ss(seq[val]), _cs(ctx[val]), torch.from_numpy(y[val])
    seq_te, ctx_te = _ss(seq[te]), _cs(ctx[te])

    model = PlayerLSTM(n_seq_f, n_ctx_f, hidden_size=hidden_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    best_val, best_state, since_best, best_epoch = float("inf"), None, 0, 0
    n = len(seq_fit)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(seq_fit[idx], ctx_fit[idx]), y_fit[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(seq_val, ctx_val), y_val))
        if v < best_val - 1e-5:
            best_val, best_state, since_best, best_epoch = \
                v, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            since_best += 1
            if since_best >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)      # restore best-val weights

    model.eval()
    with torch.no_grad():
        pred_te = model(seq_te, ctx_te).numpy()

    metrics = {
        "n_samples": int(len(seq)),
        "window": WINDOW,
        "seq_features": n_seq_f,
        "ctx_features": n_ctx_f,
        "best_epoch": best_epoch,
        "mae": float(mean_absolute_error(y[te], pred_te)),
        "r2": float(r2_score(y[te], pred_te)),
    }
    if save:
        torch.save({
            "state_dict": model.state_dict(),
            "config": {"seq_features": n_seq_f, "ctx_features": n_ctx_f,
                       "hidden_size": hidden_size, "window": WINDOW,
                       "seq_cols": SEQ_FEATURES, "ctx_cols": CTX_FEATURES},
        }, LSTM_MODEL_PATH)
        joblib.dump({"seq": seq_scaler, "ctx": ctx_scaler}, LSTM_SCALER_PATH)
        metrics["saved_to"] = str(LSTM_MODEL_PATH)
    return metrics


def load_model():
    if not LSTM_MODEL_PATH.exists() or not LSTM_SCALER_PATH.exists():
        raise FileNotFoundError(
            "LSTM artifacts missing. Run scripts/build_all.py (with torch).")
    ckpt = torch.load(LSTM_MODEL_PATH, weights_only=False)
    cfg = ckpt["config"]
    model = PlayerLSTM(cfg["seq_features"], cfg["ctx_features"],
                       hidden_size=cfg["hidden_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"model": model, "scalers": joblib.load(LSTM_SCALER_PATH),
            "config": cfg}


def forecast_sequence(match_history: list[dict], upcoming: dict,
                      bundle=None) -> dict:
    """Project next-match xG.

    match_history : list of past-match feature dicts, oldest->newest. The last
                    WINDOW entries are used (keys = SEQ_FEATURES, missing -> 0).
    upcoming      : known pre-match context for the target fixture
                    (keys = CTX_FEATURES, e.g. expected minutes, rest, travel).
    """
    bundle = bundle or load_model()
    model, scalers, cfg = bundle["model"], bundle["scalers"], bundle["config"]
    window, seq_cols, ctx_cols = cfg["window"], cfg["seq_cols"], cfg["ctx_cols"]

    if len(match_history) < window:
        raise ValueError(f"Need at least {window} matches of history, "
                         f"got {len(match_history)}.")

    seq = match_history[-window:]
    seq_arr = np.array([[float(m.get(c, 0.0)) for c in seq_cols] for m in seq],
                       dtype=np.float32)
    ctx_arr = np.array([[float(upcoming.get(c, 0.0)) for c in ctx_cols]],
                       dtype=np.float32)

    seq_s = scalers["seq"].transform(seq_arr).astype(np.float32)
    ctx_s = scalers["ctx"].transform(ctx_arr).astype(np.float32)
    with torch.no_grad():
        pred = float(model(torch.from_numpy(seq_s).unsqueeze(0),
                           torch.from_numpy(ctx_s)).item())
    return {"predicted_xg": round(max(0.0, pred), 4), "window": window,
            "matches_used": len(seq)}


if __name__ == "__main__":
    import json
    print(json.dumps(train(), indent=2))
