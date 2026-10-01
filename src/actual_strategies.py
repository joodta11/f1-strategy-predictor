"""Extract the strategies teams actually ran and compare them with the optimum."""
import pandas as pd

from src.simulator import find_optimal_strategy, simulate_strategy, strategy_label


def actual_strategy(session, driver: str):
    """A driver's real strategy in simulator format, e.g. [("MEDIUM", 0), ("HARD", 22)]."""
    laps = session.laps.pick_drivers(driver).sort_values("LapNumber")
    strategy = []
    for _, stint in laps.groupby("Stint", sort=True):
        compound = stint.Compound.mode().iat[0] if stint.Compound.notna().any() else None
        pit_lap = int(stint.LapNumber.min()) - 1
        strategy.append((compound, pit_lap))
    return strategy


def actual_strategies(session, top_n: int = 10) -> pd.DataFrame:
    """Strategies of the top-N classified finishers who completed the full distance."""
    total_laps = int(session.laps.LapNumber.max())
    results = session.results.sort_values("Position")
    rows = []
    for _, r in results.head(top_n).iterrows():
        drv = r.Abbreviation
        laps_done = session.laps.pick_drivers(drv).LapNumber.max()
        rows.append({"position": int(r.Position), "driver": drv, "team": r.TeamName,
                     "strategy": actual_strategy(session, drv),
                     "full_distance": laps_done == total_laps})
    return pd.DataFrame(rows)


def neutralised_laps(session) -> set:
    """Lap numbers on which any driver saw SC, VSC or red flag status."""
    laps = session.laps
    status = laps.TrackStatus.fillna("").astype(str)
    return set(laps.LapNumber[status.str.contains("[4567]")].astype(int))


def compare_to_actual(race_context, session, top_n: int = 10, max_stops: int = 2) -> pd.DataFrame:
    """Simulate each top-N finisher's real strategy and compare it with our optimum.

    The real races are simulated as if fully green and run by a representative
    driver, so the delta isolates the tyre/pit-timing choice, not driver pace.
    """
    ranked = find_optimal_strategy(race_context, max_stops=max_stops)
    best = ranked.iloc[0]
    sc_laps = neutralised_laps(session)
    rows = []
    for _, a in actual_strategies(session, top_n).iterrows():
        strat = a.strategy
        # Lapped drivers are simulated over the full distance, extending their last stint.
        valid = all(c in race_context.compounds for c, _ in strat)
        sim = simulate_strategy(strat, race_context)[0] if valid else float("nan")
        rows.append({
            "pos": a.position, "driver": a.driver, "team": a.team,
            "actual": strategy_label(strat), "actual_strategy": strat,
            "sim_actual_s": sim,
            "optimal": best.label, "optimal_strategy": best.strategy,
            "sim_optimal_s": best.total_time,
            "delta_s": sim - best.total_time,
            "lapped": not a.full_distance,
            "sc_stops": [p for _, p in strat[1:] if p in sc_laps],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import logging

    from src.config import RACES
    from src.data_loader import load_race
    from src.model_api import fit_race_context

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    pd.set_option("display.width", 140)
    for year, gp, _ in RACES:
        session = load_race(year, gp)
        ctx = fit_race_context(year, gp, session=session)
        cmp = compare_to_actual(ctx, session)
        print(f"\n{year} {gp} - optimal {cmp.optimal.iat[0]}, pit loss {ctx.pit_loss:.1f}s")
        print(cmp[["pos", "driver", "actual", "sim_actual_s", "delta_s", "lapped", "sc_stops"]]
              .round(1).to_string(index=False))
