"""Race strategy simulator.

A strategy is a list of stints: [("MEDIUM", 0), ("HARD", 22)] means start on
mediums and pit at the end of lap 22 for hards (lap 23 is the first hard lap).
Tyres are assumed fresh at the start of every stint (tyre age 1 on its first lap).
"""
from itertools import combinations, product

import numpy as np
import pandas as pd

from src.model_api import RaceContext, predict_lap_times


# ------------------------------------------------------------- one strategy --

def _stint_bounds(strategy, total_laps):
    """Yield (compound, first_lap, last_lap) for each stint, laps 1-indexed."""
    for i, (compound, pit_lap) in enumerate(strategy):
        first = pit_lap + 1
        last = strategy[i + 1][1] if i + 1 < len(strategy) else total_laps
        yield compound, first, last


def simulate_strategy(strategy, race_context: RaceContext):
    """Return (total race time in seconds, list of per-lap times including pit loss)."""
    ctx = race_context
    lap_times = []
    for compound, first, last in _stint_bounds(strategy, ctx.total_laps):
        laps = np.arange(first, last + 1)
        lap_times.extend(predict_lap_times(compound, laps - first + 1, laps, ctx))
    lap_times = np.asarray(lap_times, dtype=float)
    # Charge each stop's loss to the in-lap, so cumulative-gap plots show the stop.
    for _, pit_lap in strategy[1:]:
        lap_times[pit_lap - 1] += ctx.pit_loss
    return float(lap_times.sum()), lap_times.tolist()


# ------------------------------------------------------- strategy generation --

def generate_strategies(total_laps, compounds, max_stops=2, min_stint=8, max_stint=None):
    """Every valid 1..max_stops strategy using at least two different dry compounds.

    max_stint optionally caps stint length per compound (e.g. the longest stint
    actually run in the race), since the model shouldn't be trusted beyond it.
    """
    compounds = list(compounds)
    strategies = []
    for stops in range(1, max_stops + 1):
        # Pit laps p1 < p2 < ...; stint lengths are the gaps between them.
        for pits in combinations(range(min_stint, total_laps - min_stint + 1), stops):
            edges = (0, *pits, total_laps)
            lengths = np.diff(edges)
            if lengths.min() < min_stint:
                continue
            for combo in product(compounds, repeat=stops + 1):
                if len(set(combo)) < 2:
                    continue
                if max_stint and any(n > max_stint.get(c, total_laps)
                                     for c, n in zip(combo, lengths)):
                    continue
                strategies.append(list(zip(combo, edges[:-1])))
    return strategies


# ---------------------------------------------------------------- search -----

class StintTable:
    """Precomputed stint times: table[compound][first_lap, length] in seconds.

    Every strategy is then a handful of lookups instead of a lap-by-lap sum.
    """

    def __init__(self, ctx: RaceContext):
        n = ctx.total_laps
        first, length = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
        self.tables = {}
        for c in ctx.compounds:
            # lap_time[first, age] for a stint starting on `first`, valid where it ends by lap n
            laps = first + length - 1
            t = predict_lap_times(c, length.ravel(), laps.ravel(), ctx).reshape(n, n)
            t[laps > n] = np.nan
            cum = np.cumsum(t, axis=1)  # cum[first-1, L-1] = time of a stint of length L
            self.tables[c] = cum

    def stint_time(self, compound, first_lap, length):
        return self.tables[compound][first_lap - 1, length - 1]


def _table_for(ctx: RaceContext) -> StintTable:
    """Build once per context; the table doesn't depend on pit loss."""
    if getattr(ctx, "_stint_table", None) is None:
        ctx._stint_table = StintTable(ctx)
    return ctx._stint_table


def strategy_label(strategy) -> str:
    """e.g. 'M-H (pit 22)'."""
    comps = "-".join(c[0] for c, _ in strategy)
    pits = ", ".join(str(p) for _, p in strategy[1:])
    return f"{comps} (pit {pits})"


def find_optimal_strategy(race_context: RaceContext, max_stops=2, min_stint=8,
                          respect_max_stint=True) -> pd.DataFrame:
    """Simulate every valid strategy and return them ranked by total race time."""
    ctx = race_context
    table = _table_for(ctx)
    max_stint = ctx.max_tyre_age if respect_max_stint else None
    rows = []
    for strat in generate_strategies(ctx.total_laps, ctx.compounds, max_stops, min_stint,
                                     max_stint):
        total = 0.0
        for compound, first, last in _stint_bounds(strat, ctx.total_laps):
            total += table.stint_time(compound, first, last - first + 1)
        stops = len(strat) - 1
        total += stops * ctx.pit_loss
        rows.append({"strategy": strat, "label": strategy_label(strat), "stops": stops,
                     "total_time": total})
    out = pd.DataFrame(rows).sort_values("total_time", kind="stable").reset_index(drop=True)
    out["gap_to_best"] = out.total_time - out.total_time.iloc[0]
    return out


def distinct_strategies(ranked: pd.DataFrame) -> pd.DataFrame:
    """Keep only the best-timed strategy for each set of compounds.

    The raw ranking is dominated by near-copies of the winner: the same plan
    pitting a lap earlier or later, or the same stints in a different order
    (which take identical time with an additive fuel term).
    """
    keys = ranked.strategy.map(lambda s: tuple(sorted(c for c, _ in s)))
    return ranked[~keys.duplicated()].reset_index(drop=True)
