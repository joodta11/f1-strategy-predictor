"""Print a quick summary of each configured race and confirm it was dry."""
import logging

from src.config import DRY_COMPOUNDS, RACES
from src.data_loader import load_race


def summarise(session) -> dict:
    laps = session.laps
    compounds = sorted(laps["Compound"].dropna().unique())
    stints_per_driver = laps.groupby("Driver")["Stint"].max()
    return {
        "laps": int(laps["LapNumber"].max()),
        "compounds": compounds,
        "pit_stops": int((stints_per_driver - 1).clip(lower=0).sum()),
        "dry": set(compounds) <= set(DRY_COMPOUNDS) and not session.weather_data["Rainfall"].any(),
    }


if __name__ == "__main__":
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    for year, gp, character in RACES:
        s = summarise(load_race(year, gp))
        print(f"{year} {gp:8s} ({character:6s} deg): {s['laps']} laps, compounds {s['compounds']}, "
              f"{s['pit_stops']} pit stops, dry={s['dry']}")
