"""ApexSports Analytics — Streamlit dashboard.

Run:  streamlit run apexsports/dashboard/app.py

Five tabs mirroring the feature matrix: live tournament overview, xG pitch
explorer, Poisson player-goal markets, performance forecasting, and the
in-game substitution optimizer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow `streamlit run apexsports/dashboard/app.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import PITCH_LENGTH, PITCH_WIDTH, GOAL_Y
from apexsports.data.database import get_session
from apexsports.data.schema import Team, Player, Match, Shot, PlayerMatchStat
from apexsports.models import xg, poisson, forecast, calibration
from apexsports.sim.montecarlo import TeamState, simulate, optimize_substitution

st.set_page_config(page_title="ApexSports Analytics", page_icon="⚽", layout="wide")


# --- Cached data / model loaders -----------------------------------------
@st.cache_data(show_spinner=False)
def load_tables():
    with get_session() as s:
        teams = pd.read_sql(s.query(Team).statement, s.bind)
        players = pd.read_sql(s.query(Player).statement, s.bind)
        matches = pd.read_sql(s.query(Match).statement, s.bind)
        shots = pd.read_sql(s.query(Shot).statement, s.bind)
        stats = pd.read_sql(s.query(PlayerMatchStat).statement, s.bind)
    return teams, players, matches, shots, stats


@st.cache_resource(show_spinner=False)
def load_models():
    return xg.load_model(), forecast.load_model(), poisson._load_ratings()


@st.cache_resource(show_spinner=False)
def load_lstm():
    """Optional — returns the LSTM bundle, or None if torch/artifact absent."""
    try:
        from apexsports.models import lstm_forecast
        return lstm_forecast, lstm_forecast.load_model()
    except Exception:
        return None, None


def _draw_pitch(fig: go.Figure):
    """Overlay a half-pitch attacking the right-hand goal (StatsBomb coords)."""
    fig.add_shape(type="rect", x0=60, y0=0, x1=120, y1=80,
                  line=dict(color="#3a3a3a"))
    fig.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62,
                  line=dict(color="#3a3a3a"))
    fig.add_shape(type="rect", x0=114, y0=30, x1=120, y1=50,
                  line=dict(color="#3a3a3a"))
    fig.add_shape(type="line", x0=120, y0=GOAL_Y - 4, x1=120, y1=GOAL_Y + 4,
                  line=dict(color="#e63946", width=4))
    fig.update_xaxes(range=[58, 122], showgrid=False, visible=False)
    fig.update_yaxes(range=[-2, 82], showgrid=False, visible=False,
                     scaleanchor="x", scaleratio=1)


try:
    TEAMS, PLAYERS, MATCHES, SHOTS, STATS = load_tables()
    XG_MODEL, FORECAST_BUNDLE, POIS = load_models()
except Exception as e:  # pragma: no cover - guidance for first run
    st.error(f"Data/models not ready: {e}\n\nRun `python scripts/build_all.py` first.")
    st.stop()

TEAM_NAME = dict(zip(TEAMS.id, TEAMS.name))

st.title("⚽ ApexSports Analytics")
st.caption("Live, tournament-driven predictive insights — 2026 FIFA World Cup")

tab_live, tab_xg, tab_pois, tab_fc, tab_sim, tab_cal = st.tabs(
    ["📊 Tournament", "🎯 xG Explorer", "🎲 Player Goals", "📈 Forecast",
     "🔄 In-Game Sim", "📐 Calibration"])

# === Tournament overview =================================================
with tab_live:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Teams", len(TEAMS))
    c2.metric("Matches", len(MATCHES))
    c3.metric("Shots tracked", len(SHOTS))
    c4.metric("Avg xG / shot", f"{SHOTS.is_goal.mean():.3f}")

    st.subheader("Top scorers (by xG)")
    sg = STATS.groupby("player_id").agg(
        goals=("goals", "sum"), xg=("xg", "sum"),
        minutes=("minutes", "sum")).reset_index()
    sg = sg.merge(PLAYERS[["id", "name", "team_id", "position"]],
                  left_on="player_id", right_on="id")
    sg["team"] = sg.team_id.map(TEAM_NAME)
    top = sg.sort_values("xg", ascending=False).head(15)
    fig = px.bar(top, x="xg", y="name", orientation="h", color="goals",
                 hover_data=["team", "goals", "minutes"],
                 color_continuous_scale="Tealgrn",
                 labels={"xg": "Expected Goals", "name": ""})
    fig.update_layout(height=480, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent fixtures")
    md = MATCHES.copy()
    md["home"] = md.home_team_id.map(TEAM_NAME)
    md["away"] = md.away_team_id.map(TEAM_NAME)
    md["result"] = md.home + " " + md.home_goals.astype(str) + " - " + \
        md.away_goals.astype(str) + " " + md.away
    st.dataframe(md[["date", "stage", "city", "result"]]
                 .sort_values("date", ascending=False).head(12),
                 use_container_width=True, hide_index=True)

# === xG explorer ==========================================================
with tab_xg:
    st.subheader("Expected Goals — shot-location explorer")
    left, right = st.columns([1, 2])
    with left:
        x = st.slider("Distance axis (x)", 60.0, 119.0, 105.0, 0.5)
        y = st.slider("Width axis (y)", 0.0, 80.0, 40.0, 0.5)
        header = st.checkbox("Header")
        pressure = st.checkbox("Under pressure")
        big = st.checkbox("Big chance")
        res = xg.predict_xg(x, y, header, pressure, big, model=XG_MODEL)
        st.metric("xG for this shot", f"{res['xg']:.3f}")
        st.write(f"Distance to goal: **{res['distance']:.1f}** units")
        st.write(f"Goal-mouth angle: **{np.degrees(res['angle_rad']):.1f}°**")

    with right:
        # xG surface heatmap across the attacking third.
        xs = np.linspace(80, 119, 40)
        ys = np.linspace(8, 72, 40)
        grid = np.array([[xg.predict_xg(xi, yi, header, pressure, big,
                                        model=XG_MODEL)["xg"]
                          for xi in xs] for yi in ys])
        fig = go.Figure(go.Heatmap(x=xs, y=ys, z=grid, colorscale="YlOrRd",
                                   colorbar=dict(title="xG")))
        _draw_pitch(fig)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers",
                                 marker=dict(size=16, color="cyan",
                                             line=dict(color="black", width=2)),
                                 name="Your shot"))
        fig.update_layout(height=480, title="xG surface (attacking third)")
        st.plotly_chart(fig, use_container_width=True)

# === Poisson player goals =================================================
with tab_pois:
    st.subheader("Player goal distribution (Poisson)")
    c1, c2, c3 = st.columns(3)
    team_sel = c1.selectbox("Team", TEAMS.name.tolist(), key="pois_team")
    tid = int(TEAMS.loc[TEAMS.name == team_sel, "id"].iloc[0])
    squad = PLAYERS[(PLAYERS.team_id == tid) &
                    (PLAYERS.position.isin(["FWD", "MID"]))]
    pl_sel = c2.selectbox("Player", squad.name.tolist(), key="pois_player")
    pid = int(squad.loc[squad.name == pl_sel, "id"].iloc[0])
    opp_sel = c3.selectbox("Opponent", [t for t in TEAMS.name if t != team_sel],
                           key="pois_opp")
    oid = int(TEAMS.loc[TEAMS.name == opp_sel, "id"].iloc[0])
    minutes = st.slider("Expected minutes", 15, 120, 90, 5, key="pois_min")

    dist = poisson.player_goal_distribution(pid, oid, minutes, ratings=POIS)
    m1, m2, m3 = st.columns(3)
    m1.metric("λ (expected goals)", f"{dist['lambda']:.2f}")
    m2.metric("P(scores ≥ 1)", f"{dist['p_at_least_1']:.1%}")
    m3.metric("P(brace+)", f"{dist['p_brace_plus']:.1%}")

    dd = pd.DataFrame({"goals": list(dist["distribution"].keys()),
                       "prob": list(dist["distribution"].values())})
    fig = px.bar(dd, x="goals", y="prob", text_auto=".1%",
                 labels={"goals": "Goals", "prob": "Probability"},
                 color="prob", color_continuous_scale="Blues")
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"vs {opp_sel} (defence factor {dist['opponent_defence_factor']:.2f}) "
               f"— higher factor = leakier defence ⇒ higher λ.")

# === Forecast =============================================================
with tab_fc:
    st.subheader("Performance forecast — upcoming fixture (XGBoost)")
    st.caption("Adjust tournament-context inputs to see projected xG output. "
               "Note: single-match xG is intrinsically noisy; the model captures "
               "directional effects of fatigue, rest and travel.")
    c1, c2, c3 = st.columns(3)
    skill = c1.slider("Player skill", 0.0, 1.0, 0.7, 0.05)
    pos = c1.selectbox("Position", ["FWD", "MID", "DEF", "GK"])
    pos_code = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}[pos]
    rest = c2.slider("Rest days", 2, 8, 4)
    travel = c2.slider("Travel km (since last match)", 0, 4000, 800, 50)
    elevation = c3.slider("Stadium elevation (m)", 0, 2240, 100, 10)
    fatigue = c3.slider("Fatigue index", 0.0, 1.0, 0.3, 0.05)

    feats = {"skill": skill, "position_code": pos_code, "rest_days": rest,
             "travel_km": travel, "elevation_m": elevation,
             "fatigue_index": fatigue, "form_xg3": 0.25 * skill,
             "form_minutes3": 80.0, "career_xg90": 0.45 * skill}
    out = forecast.forecast_player(feats, bundle=FORECAST_BUNDLE)

    # Optional LSTM sequence forecast for side-by-side comparison.
    lstm_mod, lstm_bundle = load_lstm()
    mcol1, mcol2 = st.columns(2)
    mcol1.metric("XGBoost — projected xG", f"{out['predicted_xg']:.3f}")
    if lstm_bundle is not None:
        window = lstm_bundle["config"]["window"]
        history = [{"minutes": feats["form_minutes3"], "xg": feats["form_xg3"],
                    "goals": 0, "shots": 2, "passes": 30, "distance_km": 9.5,
                    "assists": 0, "rest_days": rest, "travel_km": travel,
                    "elevation_m": elevation, "fatigue_index": fatigue,
                    "skill": skill, "position_code": pos_code}
                   for _ in range(window)]
        upcoming = {"minutes": 90, "rest_days": rest, "travel_km": travel,
                    "elevation_m": elevation, "fatigue_index": fatigue,
                    "skill": skill, "position_code": pos_code}
        lstm_out = lstm_mod.forecast_sequence(history, upcoming, bundle=lstm_bundle)
        mcol2.metric("LSTM (sequence) — projected xG",
                     f"{lstm_out['predicted_xg']:.3f}")
    else:
        mcol2.info("LSTM unavailable (install torch + run build_all).")

    # Sensitivity sweep over fatigue.
    fr = np.linspace(0, 1, 25)
    proj = [forecast.forecast_player({**feats, "fatigue_index": f},
                                     bundle=FORECAST_BUNDLE)["predicted_xg"]
            for f in fr]
    fig = px.line(x=fr, y=proj, labels={"x": "Fatigue index", "y": "Projected xG"},
                  title="Projected output vs fatigue")
    fig.add_vline(x=fatigue, line_dash="dash", line_color="red")
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

# === In-game simulation ===================================================
with tab_sim:
    st.subheader("In-game scenario & substitution optimizer")
    st.caption("Monte Carlo the remaining minutes and find the mentality switch "
               "that best serves your objective (e.g. protect a 1-0 lead).")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Home team**")
        h_atk = st.slider("Home attack (goals/90)", 0.5, 2.5, 1.5, 0.05)
        h_def = st.slider("Home defence strength", 0.8, 1.6, 1.25, 0.05)
        h_fat = st.slider("Home fatigue", 0.0, 1.0, 0.5, 0.05)
    with c2:
        st.markdown("**Away team**")
        a_atk = st.slider("Away attack (goals/90)", 0.5, 2.5, 1.1, 0.05)
        a_def = st.slider("Away defence strength", 0.8, 1.6, 1.15, 0.05)
        a_fat = st.slider("Away fatigue", 0.0, 1.0, 0.4, 0.05)

    c3, c4, c5, c6 = st.columns(4)
    minute = c3.slider("Current minute", 0, 89, 75)
    hg = c4.number_input("Home goals", 0, 9, 1)
    ag = c5.number_input("Away goals", 0, 9, 0)
    objective = c6.selectbox("Objective", ["hold", "win", "comeback"])

    home = TeamState("Home", h_atk, h_def, h_fat)
    away = TeamState("Away", a_atk, a_def, a_fat)

    base = simulate(home, away, minute, hg, ag, n_sims=15000)
    o1, o2, o3 = st.columns(3)
    o1.metric("P(Home win)", f"{base['home_win']:.1%}")
    o2.metric("P(Draw)", f"{base['draw']:.1%}")
    o3.metric("P(Away win)", f"{base['away_win']:.1%}")

    rec = optimize_substitution(home, away, minute, hg, ag, objective)
    st.success(f"**Recommendation: switch to '{rec['recommendation']}'** — "
               f"{rec['rationale']}")

    opt_df = pd.DataFrame(rec["options"])
    fig = px.bar(opt_df, x="mentality", y=["home_win", "draw", "away_win"],
                 barmode="stack", title="Outcome probabilities by mentality",
                 labels={"value": "Probability", "mentality": "Mentality"})
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Most likely final scorelines")
    st.dataframe(pd.DataFrame(base["top_scorelines"]),
                 use_container_width=True, hide_index=True)

# === xG calibration: our model vs StatsBomb ==============================
with tab_cal:
    st.subheader("xG calibration — our model vs StatsBomb")
    st.caption("Reliability diagram: each point is a probability bin; x is the "
               "mean predicted xG, y the observed goal rate. Points on the "
               "dashed diagonal are perfectly calibrated.")

    cal = calibration.compare(n_bins=10, model=XG_MODEL)
    st.write(f"Evaluated on **{cal['n_shots']:,} shots**.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"),
                             name="Perfect calibration"))

    def _add_curve(curve, name, color):
        d = pd.DataFrame(curve)
        fig.add_trace(go.Scatter(
            x=d["mean_predicted"], y=d["observed_freq"], mode="lines+markers",
            name=name, line=dict(color=color),
            marker=dict(size=(d["count"].clip(upper=400) / 18 + 6)),
            hovertext=[f"n={c}" for c in d["count"]]))

    _add_curve(cal["our"]["curve"], "Our xG (logistic)", "#1f77b4")
    if cal["has_statsbomb"]:
        _add_curve(cal["statsbomb"]["curve"], "StatsBomb xG", "#e63946")
    fig.update_layout(height=460, xaxis_title="Mean predicted xG",
                      yaxis_title="Observed goal frequency",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

    if cal["has_statsbomb"]:
        sc = pd.DataFrame({
            "Metric": ["Brier (↓)", "Log loss (↓)", "AUC (↑)", "Mean xG"],
            "Our model": [cal["our"]["score"]["brier"],
                          cal["our"]["score"]["log_loss"],
                          cal["our"]["score"].get("auc", float("nan")),
                          cal["our"]["score"]["mean_xg"]],
            "StatsBomb": [cal["statsbomb"]["score"]["brier"],
                          cal["statsbomb"]["score"]["log_loss"],
                          cal["statsbomb"]["score"].get("auc", float("nan")),
                          cal["statsbomb"]["score"]["mean_xg"]],
        }).round(4)
        st.dataframe(sc, use_container_width=True, hide_index=True)
        a1, a2 = st.columns(2)
        a1.metric("Shot-for-shot correlation (r)",
                  f"{cal['agreement']['pearson_r']:.3f}")
        a2.metric("Mean |our − StatsBomb|",
                  f"{cal['agreement']['mean_abs_diff']:.3f}")
        st.caption(f"Actual conversion rate: "
                   f"{cal['our']['score']['actual_rate']:.1%}. Our 5-feature "
                   "logistic model tracks StatsBomb's full xG closely.")
    else:
        st.info("StatsBomb reference xG is only available on real data. "
                "Load it with `python scripts/build_all.py --source statsbomb`, "
                "then reopen this tab. (Synthetic data has no reference xG.)")

