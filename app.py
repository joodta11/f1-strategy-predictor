"""Streamlit front end for the F1 tyre strategy predictor.

Run from the project root:  streamlit run app.py
"""
import dataclasses
import logging

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.actual_strategies import compare_to_actual
from src.cleaning import clean_laps
from src.config import RACES
from src.data_loader import load_race
from src.model_api import fit_race_context
from src.plots import (plot_actual_vs_optimal, plot_degradation, plot_pit_window,
                       plot_strategy_comparison)
from src.simulator import distinct_strategies, find_optimal_strategy

logging.getLogger("fastf1").setLevel(logging.ERROR)
st.set_page_config(page_title="F1 Strategy Predictor", layout="wide")


# FastF1 sessions and fitted contexts aren't reliably picklable, so they are cached
# as resources; the derived tables are plain DataFrames and use cache_data.
@st.cache_resource(show_spinner="Loading race data (first run downloads from FastF1)...")
def load(year, gp):
    session = load_race(year, gp)
    return session, fit_race_context(year, gp, session=session), clean_laps(session)


@st.cache_data(show_spinner="Simulating every strategy...")
def ranked_strategies(year, gp, max_stops, pit_loss):
    _, ctx, _ = load(year, gp)
    return find_optimal_strategy(dataclasses.replace(ctx, pit_loss=pit_loss), max_stops=max_stops)


@st.cache_data(show_spinner="Comparing with what teams actually did...")
def comparison(year, gp, max_stops, pit_loss):
    session, ctx, _ = load(year, gp)
    return compare_to_actual(dataclasses.replace(ctx, pit_loss=pit_loss), session,
                             max_stops=max_stops)


def fmt_time(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:06.3f}"


def explain(best, ranked, ctx):
    stints = best.strategy
    parts = [f"start on **{stints[0][0].title()}**"]
    parts += [f"pit on lap **{p}** for **{c.title()}**" for c, p in stints[1:]]
    text = "Plan: " + ", then ".join(parts) + "."
    other = ranked[ranked.stops != best.stops]
    if len(other):
        alt = other.iloc[0]
        word = {1: "one-stop", 2: "two-stop"}
        text += (f" The best {word[alt.stops]} ({alt.label}) is **{alt.gap_to_best:.1f}s slower**. "
                 f"Each stop costs about {ctx.pit_loss:.1f}s, and degradation here runs at ")
        text += ", ".join(f"{c.title().lower()} {ctx.deg_rate[c]:.3f}s/lap" for c in ctx.compounds)
        text += ("; " + ("the time saved on fresher tyres outweighs the extra stop."
                         if best.stops > alt.stops else
                         "fresher tyres don't win back enough time to pay for another stop."))
    return text


def show(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ---------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.header("Race")
    labels = {f"{y} {gp} GP ({deg} degradation)": (y, gp) for y, gp, deg in RACES}
    year, gp = labels[st.selectbox("Race", list(labels))]
    max_stops = st.radio("Max stops", [1, 2], index=1, horizontal=True)
    session, ctx, laps = load(year, gp)
    pit_loss = st.slider("Pit loss (s)", 15.0, 35.0, float(round(ctx.pit_loss, 1)), 0.1,
                         help=f"Estimated from real green-flag stops: {ctx.pit_loss:.1f}s")
    st.caption("Model: linear degradation fitted on cleaned green-flag laps. "
               "No safety cars, traffic or tyre warm-up are modelled.")

run_ctx = dataclasses.replace(ctx, pit_loss=pit_loss)
ranked = ranked_strategies(year, gp, max_stops, pit_loss)
cmp = comparison(year, gp, max_stops, pit_loss)
best = ranked.iloc[0]

# ------------------------------------------------------------------- main ----
st.title(f"{year} {gp} GP: strategy predictor")

c1, c2, c3 = st.columns(3)
c1.metric("Recommended strategy", best.label)
c2.metric("Predicted race time", fmt_time(best.total_time))
c3.metric("Strategies simulated", f"{len(ranked):,}")
st.markdown(explain(best, ranked, run_ctx))

st.subheader("Top 5 strategies")
top = distinct_strategies(ranked).head(5)
st.dataframe(pd.DataFrame({
    "Strategy": top.label, "Stops": top.stops,
    "Race time": top.total_time.map(fmt_time),
    "Gap to best (s)": top.gap_to_best.round(2),
}), hide_index=True, width="stretch")
st.caption("Best pit timing for each set of compounds; the same stints in a different order "
           "take the same time in this model.")

left, right = st.columns(2)
with left:
    show(plot_degradation(run_ctx, laps, save=False))
with right:
    show(plot_pit_window(run_ctx, ranked, save=False))
show(plot_strategy_comparison(run_ctx, ranked, save=False))
show(plot_actual_vs_optimal(run_ctx, cmp, save=False))

st.subheader("Top 10: actual vs optimal")
st.dataframe(pd.DataFrame({
    "Pos": cmp.pos, "Driver": cmp.driver, "Team": cmp.team,
    "Actual strategy": cmp.actual,
    "Simulated time": cmp.sim_actual_s.map(fmt_time),
    "Optimal": cmp.optimal,
    "Delta (s)": cmp.delta_s.round(1),
    "Notes": [", ".join(n for n, on in [("stop under SC", bool(sc)), ("lapped", lap)] if on)
              for sc, lap in zip(cmp.sc_stops, cmp.lapped)],
}), hide_index=True, width="stretch")
st.caption("Simulated green-flag times for a median-pace driver, so the delta reflects the "
           "strategy choice, not driver pace. Stops under safety car were cheaper in reality.")
