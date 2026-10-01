"""Load F1 race sessions through FastF1 with a local on-disk cache."""
from pathlib import Path

import fastf1

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

_cache_enabled = False


def enable_cache() -> None:
    global _cache_enabled
    if not _cache_enabled:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        _cache_enabled = True


def load_race(year: int, gp_name: str):
    """Load the race session (laps, weather, track status) for one Grand Prix."""
    enable_cache()
    session = fastf1.get_session(year, gp_name, "R")
    session.load(laps=True, weather=True, messages=True, telemetry=False)
    return session
