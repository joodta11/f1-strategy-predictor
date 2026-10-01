"""Exploratory plots: lap time vs tyre age and vs lap number for each race.

Run from the project root:  python -m notebooks.eda
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cleaning import clean_laps
from src.config import RACES
from src.data_loader import load_race

OUT = Path(__file__).resolve().parent.parent / "figures" / "eda"
COLOURS = {"SOFT": "#E8002D", "MEDIUM": "#FFD12E", "HARD": "#9A9A9A"}


def plot_race(year, gp, laps):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Remove driver pace differences so the tyre and fuel trends are visible.
    laps = laps.assign(rel=laps.lap_time_s - laps.groupby("driver").lap_time_s.transform("median"))
    for comp, g in laps.groupby("compound"):
        c = COLOURS[comp]
        axes[0].scatter(g.tyre_age, g.rel, s=8, alpha=0.35, color=c,
                        edgecolors="k" if comp == "HARD" else "none", linewidths=0.2, label=comp)
        med = g.groupby("tyre_age").rel.median()
        axes[0].plot(med.index, med.values, color=c, lw=2.5,
                     path_effects=None)
        axes[1].scatter(g.lap_number, g.rel, s=8, alpha=0.35, color=c,
                        edgecolors="k" if comp == "HARD" else "none", linewidths=0.2, label=comp)
    axes[0].set(xlabel="Tyre age (laps)", ylabel="Lap time vs driver median (s)",
                title="Lap time vs tyre age")
    axes[1].set(xlabel="Lap number", ylabel="Lap time vs driver median (s)",
                title="Lap time vs lap number")
    for ax in axes:
        ax.set_ylim(-3, 3)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{year} {gp} GP - cleaned green-flag laps")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{year}_{gp.lower().replace(' ', '_')}_eda.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def describe(laps):
    """Per-compound slope vs tyre age (fuel-corrected) and curvature, for the write-up."""
    laps = laps.assign(rel=laps.lap_time_s - laps.groupby("driver").lap_time_s.transform("median"))
    fuel = np.polyfit(laps.lap_number, laps.rel, 1)[0]
    print(f"  raw slope vs lap number: {fuel:+.3f} s/lap")
    for comp, g in laps.groupby("compound"):
        X = np.column_stack([g.tyre_age, g.tyre_age ** 2, g.lap_number, np.ones(len(g))])
        coef, *_ = np.linalg.lstsq(X, g.rel, rcond=None)
        print(f"  {comp:6s} n={len(g):4d} linear={coef[0]:+.3f} quad={coef[1]:+.4f} fuel={coef[2]:+.3f}")


if __name__ == "__main__":
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    for year, gp, _ in RACES:
        laps = clean_laps(load_race(year, gp))
        print(f"{gp}: saved {plot_race(year, gp, laps)}")
        describe(laps)
