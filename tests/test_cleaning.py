from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.cleaning import clean_laps


def _fake_session():
    """Two drivers, 12 laps each, with a pit stop, an SC period and junk laps."""
    rows = []
    for drv, base in [("AAA", 90.0), ("BBB", 90.5)]:
        for lap in range(1, 13):
            stint = 1 if lap <= 6 else 2
            tyre_life = lap if stint == 1 else lap - 6
            rows.append({
                "Driver": drv, "Team": "Team" + drv, "LapNumber": lap,
                "LapTime": pd.Timedelta(seconds=base + 0.05 * tyre_life),
                "LapStartTime": pd.Timedelta(minutes=lap * 1.5),
                "Compound": "MEDIUM" if stint == 1 else "HARD",
                "TyreLife": float(tyre_life), "Stint": float(stint),
                "PitInTime": pd.Timedelta(minutes=9) if lap == 6 else pd.NaT,
                "PitOutTime": pd.Timedelta(minutes=9.1) if lap == 7 else pd.NaT,
                "TrackStatus": "14" if lap in (9, 10) else "1",
                "Deleted": False, "IsAccurate": True,
            })
    laps = pd.DataFrame(rows)
    laps.loc[(laps.Driver == "AAA") & (laps.LapNumber == 4), "LapTime"] = pd.Timedelta(seconds=130)
    laps.loc[(laps.Driver == "BBB") & (laps.LapNumber == 3), "Deleted"] = True
    laps.loc[(laps.Driver == "BBB") & (laps.LapNumber == 5), "Compound"] = "INTERMEDIATE"
    weather = pd.DataFrame({"Time": pd.to_timedelta(np.arange(0, 30), unit="min"),
                            "TrackTemp": np.linspace(40, 35, 30)})
    return SimpleNamespace(laps=laps, weather_data=weather)


def test_no_pit_laps_remain():
    out = clean_laps(_fake_session())
    assert not ((out.lap_number == 6) | (out.lap_number == 7)).any()


def test_no_safety_car_laps_remain():
    out = clean_laps(_fake_session())
    assert not out.lap_number.isin([9, 10]).any()


def test_lap_one_outliers_deleted_and_wet_removed():
    out = clean_laps(_fake_session())
    assert (out.lap_number > 1).all()
    assert out.lap_time_s.max() < 100
    assert not ((out.driver == "BBB") & out.lap_number.isin([3, 5])).any()
    assert set(out.compound) <= {"SOFT", "MEDIUM", "HARD"}


def test_columns_and_track_temp():
    out = clean_laps(_fake_session())
    assert list(out.columns) == ["driver", "team", "lap_number", "lap_time_s", "compound",
                                 "tyre_age", "stint", "track_temp"]
    assert out.track_temp.between(35, 40).all()
