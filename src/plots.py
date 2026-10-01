"""Static charts for each race, saved as PNGs under figures/.

Run from the project root:  python -m src.plots
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.model_api import predict_lap_times
from src.simulator import distinct_strategies, find_optimal_strategy, simulate_strategy

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# Official Pirelli compound colours; hard is white, so it gets a dark outline.
COMPOUND_COLOURS = {"SOFT": "#DA291C", "MEDIUM": "#FFD12E", "HARD": "#FFFFFF"}
COMPOUND_EDGE = {"SOFT": "#DA291C", "MEDIUM": "#C9A000", "HARD": "#222222"}
# Line colour for the hard compound on a white background.
COMPOUND_LINE = {"SOFT": "#DA291C", "MEDIUM": "#E0B000", "HARD": "#555555"}
# Categorical order for "series" that aren't compounds (validated colour-blind safe palette).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

INK = "#1f1f1f"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#bbbbbb", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def _slug(ctx):
    return f"{ctx.year}_{ctx.gp_name.lower().replace(' ', '_')}"


def _title(ax, title, subtitle):
    ax.set_title(title, pad=10 + 13 * (subtitle.count(chr(10)) + 1))
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=MUTED, fontsize=9.5, va="bottom")


def _save(fig, name, save=True):
    """Write the PNG and return its path, or return the figure itself (for the app)."""
    if not save:
        return fig
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------ 1. degradation -------

def _spread_labels(ax, labels, min_gap_frac=0.06):
    """Direct-label line ends, nudging labels apart vertically so they never overlap."""
    ax.figure.canvas.draw()
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    labels = sorted(labels)
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + gap)
    x_text = max(x for _, x, _ in labels) + 1
    for y, x, text in labels:
        ax.annotate(text, (x, y), xytext=(x_text, y), va="center", color=INK, fontsize=9.5)

def plot_degradation(ctx, laps: pd.DataFrame | None = None, save=True):
    """Predicted lap time vs tyre age per compound, at a fixed mid-race fuel load.

    If cleaned laps are given, their fuel- and driver-corrected medians per tyre
    age are drawn as dots so the fit can be checked by eye.
    """
    ref_lap = ctx.total_laps // 2
    offs = None
    if laps is not None:
        from src.models import BaselineModel
        offs = BaselineModel().fit(laps).driver_offsets()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    labels = []
    for c in ctx.compounds:
        ages = np.arange(1, ctx.max_tyre_age[c] + 1)
        pred = predict_lap_times(c, ages, ref_lap, ctx)
        ax.plot(ages, pred, color=COMPOUND_LINE[c], lw=2.2, zorder=3)
        labels.append([pred[-1], ages[-1], f"{c.title()}  {ctx.deg_rate[c]:+.3f} s/lap"])
        if offs is not None:
            g = laps[laps.compound == c]
            if len(g):
                adj = (g.lap_time_s - g.driver.map(offs) + ctx.driver_offset
                       - ctx.fuel_effect * (g.lap_number - ref_lap))
                med = adj.groupby(g.tyre_age).median()
                n = g.groupby("tyre_age").size()
                med = med[n >= 3]
                ax.scatter(med.index, med.values, s=26, color=COMPOUND_COLOURS[c],
                           edgecolors=COMPOUND_EDGE[c], linewidths=1, zorder=4, alpha=0.9)
    _spread_labels(ax, labels)
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel(f"Lap time at lap {ref_lap} fuel load (s)")
    ax.set_xlim(0, max(ctx.max_tyre_age.values()) * 1.32)
    _title(ax, f"{ctx.year} {ctx.gp_name} GP: tyre degradation",
           "Lines: fitted model for a median driver. Dots: median of real laps, fuel- and driver-corrected.")
    return _save(fig, f"{_slug(ctx)}_degradation.png", save)


# --------------------------------------------- 2. strategy comparison --------

def plot_strategy_comparison(ctx, ranked: pd.DataFrame, top_n=5, save=True):
    """Cumulative time gap per lap for the top distinct strategies relative to the best."""
    top = distinct_strategies(ranked).head(top_n)
    best_laps = np.cumsum(simulate_strategy(top.strategy.iat[0], ctx)[1])
    fig, ax = plt.subplots(figsize=(10, 5.2))
    laps = np.arange(1, ctx.total_laps + 1)
    for i, (_, row) in enumerate(top.iterrows()):
        gap = np.cumsum(simulate_strategy(row.strategy, ctx)[1]) - best_laps
        ax.plot(laps, gap, color=SERIES[i], lw=2, label=f"{row.label}  +{row.gap_to_best:.1f}s",
                zorder=5 - i)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Gap to best strategy (s, + = behind)")
    ax.set_xlim(1, ctx.total_laps)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9)
    _title(ax, f"{ctx.year} {ctx.gp_name} GP: top {len(top)} strategies",
           "Best timing for each set of compounds. Jumps up are pit stops; the right edge is the final gap.")
    return _save(fig, f"{_slug(ctx)}_strategy_comparison.png", save)


# ------------------------------------------------- 3. actual vs optimal ------

def _stint_bars(ax, y, strategy, total_laps, height=0.62, gap=0.25):
    for i, (comp, pit) in enumerate(strategy):
        end = strategy[i + 1][1] if i + 1 < len(strategy) else total_laps
        # A small surface gap between stints so back-to-back stints on one compound stay distinct.
        left = pit + (gap if i else 0)
        right = end - (gap if i + 1 < len(strategy) else 0)
        ax.barh(y, right - left, left=left, height=height,
                color=COMPOUND_COLOURS.get(comp, "#cccccc"),
                edgecolor=COMPOUND_EDGE.get(comp, "#888888"), linewidth=1.2, zorder=3)
        ax.text(pit + (end - pit) / 2, y, comp[0], ha="center", va="center", fontsize=8.5,
                color="white" if comp == "SOFT" else INK, fontweight="bold", zorder=4)


def plot_actual_vs_optimal(ctx, comparison: pd.DataFrame, save=True):
    rows = [("Optimal (model)", comparison.optimal_strategy.iat[0], None, [])]
    rows += [(f"P{r.pos}  {r.driver}" + ("*" if r.lapped else ""), r.actual_strategy,
              r.delta_s, r.sc_stops) for r in comparison.itertuples()]
    fig, ax = plt.subplots(figsize=(10, 0.48 * len(rows) + 1.6))
    for y, (name, strat, delta, sc_stops) in enumerate(rows):
        _stint_bars(ax, y, strat, ctx.total_laps)
        label = "fastest" if delta is None else f"+{delta:.1f}s"
        if sc_stops:
            label += "  (SC stop)"
        ax.text(ctx.total_laps + 1, y, label, va="center", fontsize=9,
                color=INK if delta is None else MUTED)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, ctx.total_laps * 1.2)
    ax.set_xlabel("Lap")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.axhline(0.5, color="#bbbbbb", lw=1)
    handles = [Patch(facecolor=COMPOUND_COLOURS[c], edgecolor=COMPOUND_EDGE[c], label=c.title())
               for c in ("SOFT", "MEDIUM", "HARD")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / len(rows)),
              ncol=3, fontsize=9)
    _title(ax, f"{ctx.year} {ctx.gp_name} GP: actual vs optimal strategy",
           "Top 10 finishers, each plan simulated green-flag for a median driver vs the model's best.\n"
           "SC stop = a stop made under safety car, cheaper in reality than simulated. * = finished a lap down.")
    return _save(fig, f"{_slug(ctx)}_actual_vs_optimal.png", save)


# ---------------------------------------------- 4. pit window sensitivity ----

def plot_pit_window(ctx, ranked: pd.DataFrame, save=True):
    """Total race time vs pit lap for the best one-stop compound pair."""
    one = ranked[ranked.stops == 1]
    best = one.iloc[0].strategy
    pair = (best[0][0], best[1][0])
    sub = one[one.strategy.map(lambda s: (s[0][0], s[1][0]) == pair)].copy()
    sub["pit_lap"] = sub.strategy.map(lambda s: s[1][1])
    sub = sub.sort_values("pit_lap")
    gap = sub.total_time - ranked.total_time.iat[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sub.pit_lap, gap, color=SERIES[0], lw=2, marker="o", ms=4)
    i = gap.idxmin()
    ax.scatter([sub.pit_lap[i]], [gap[i]], s=80, color=SERIES[0], edgecolors="white",
               linewidths=2, zorder=5)
    ax.annotate(f"Best: pit lap {sub.pit_lap[i]}\n+{gap[i]:.1f}s vs overall best",
                (sub.pit_lap[i], gap[i]), xytext=(0, 40), textcoords="offset points",
                ha="center", fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    window = sub.pit_lap[gap <= gap[i] + 1.0]
    ax.axvspan(window.min() - 0.5, window.max() + 0.5, color=SERIES[0], alpha=0.08, lw=0)
    ax.set_xlabel("Pit lap")
    ax.set_ylabel("Race time vs best strategy overall (s)")
    _title(ax, f"{ctx.year} {ctx.gp_name} GP: pit window for {pair[0].title()} → {pair[1].title()}",
           f"Best one-stop compound order. Shaded: within 1s of the best pit lap "
           f"(laps {window.min()}-{window.max()}).")
    return _save(fig, f"{_slug(ctx)}_pit_window.png", save)


def plot_all(ctx, session, laps=None):
    from src.actual_strategies import compare_to_actual

    ranked = find_optimal_strategy(ctx, max_stops=2)
    return [
        plot_degradation(ctx, laps),
        plot_strategy_comparison(ctx, ranked),
        plot_actual_vs_optimal(ctx, compare_to_actual(ctx, session)),
        plot_pit_window(ctx, ranked),
    ]


if __name__ == "__main__":
    import logging

    from src.cleaning import clean_laps
    from src.config import RACES
    from src.data_loader import load_race
    from src.model_api import fit_race_context

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    for year, gp, _ in RACES:
        session = load_race(year, gp)
        ctx = fit_race_context(year, gp, session=session)
        for p in plot_all(ctx, session, clean_laps(session)):
            print(p.relative_to(FIG_DIR.parent))
