"""Estimate the time lost to a pit stop from real race laps."""
import numpy as np
import pandas as pd

from src.cleaning import _green_only

WINDOW = 4  # green laps either side of the stop used as the driver's reference pace


def pit_stop_losses(session) -> pd.DataFrame:
    """One row per green-flag pit stop: in-lap + out-lap compared with nearby normal laps."""
    laps = pd.DataFrame(session.laps)
    laps = laps.assign(t=laps.LapTime.dt.total_seconds(), green=_green_only(laps.TrackStatus))
    rows = []
    for drv, d in laps.groupby("Driver"):
        d = d.set_index("LapNumber").sort_index()
        normal = d[d.PitInTime.isna() & d.PitOutTime.isna() & d.green & d.t.notna()
                   & (d.index > 1)]
        for in_lap in d.index[d.PitInTime.notna()]:
            out_lap = in_lap + 1
            if out_lap not in d.index or pd.isna(d.at[out_lap, "PitOutTime"]):
                continue  # retirement or missing data
            if not (d.at[in_lap, "green"] and d.at[out_lap, "green"]):
                continue  # stops under SC/VSC are much cheaper and not representative
            pair = d.loc[[in_lap, out_lap], "t"]
            if pair.isna().any():
                continue
            nearby = normal[(normal.index >= in_lap - WINDOW) & (normal.index <= out_lap + WINDOW)].t
            if len(nearby) < 3:
                continue
            rows.append({"driver": drv, "in_lap": int(in_lap),
                         "loss": pair.sum() - 2 * nearby.median()})
    return pd.DataFrame(rows, columns=["driver", "in_lap", "loss"])


def estimate_pit_loss(session) -> float:
    """Median across drivers of each driver's average green-flag pit loss, in seconds."""
    stops = pit_stop_losses(session)
    if stops.empty:
        return float("nan")
    # Drop obviously broken stops (slow wheel change, damage repairs, drive-throughs).
    stops = stops[stops.loss.between(10, 45)]
    return float(np.median(stops.groupby("driver").loss.mean()))


if __name__ == "__main__":
    import logging

    from src.config import RACES
    from src.data_loader import load_race

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    for year, gp, _ in RACES:
        stops = pit_stop_losses(load_race(year, gp))
        print(f"{year} {gp:8s}: pit loss {estimate_pit_loss(load_race(year, gp)):.1f}s "
              f"from {len(stops)} green-flag stops")
