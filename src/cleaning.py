"""Reduce a race session to representative green-flag laps."""
import pandas as pd

from src.config import DRY_COMPOUNDS

OUTLIER_THRESHOLD = 1.07  # drop laps slower than 107% of the driver's stint median

COLUMNS = ["driver", "team", "lap_number", "lap_time_s", "compound",
           "tyre_age", "stint", "track_temp"]


def _green_only(track_status: pd.Series) -> pd.Series:
    """TrackStatus lists every status seen during the lap; '1' is the only green code.

    Anything containing 2 (yellow), 4 (SC), 5 (red), 6/7 (VSC) is excluded.
    """
    return track_status.fillna("").astype(str).str.fullmatch(r"1+")


def clean_laps(session) -> pd.DataFrame:
    laps = pd.DataFrame(session.laps).copy()

    keep = (
        (laps["LapNumber"] > 1)
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & _green_only(laps["TrackStatus"])
        & (laps["Deleted"] != True)  # noqa: E712 - column can hold NaN
        & (laps["IsAccurate"] == True)  # noqa: E712
        & laps["Compound"].isin(DRY_COMPOUNDS)
        & laps["LapTime"].notna()
        & laps["TyreLife"].notna()
    )
    laps = laps[keep].copy()
    laps["lap_time_s"] = laps["LapTime"].dt.total_seconds()

    stint_median = laps.groupby(["Driver", "Stint"])["lap_time_s"].transform("median")
    laps = laps[laps["lap_time_s"] <= OUTLIER_THRESHOLD * stint_median]

    # Track temperature from the weather feed (~1 sample/min), nearest in session time.
    weather = session.weather_data[["Time", "TrackTemp"]].sort_values("Time")
    laps = laps.sort_values("LapStartTime")
    laps = pd.merge_asof(laps, weather, left_on="LapStartTime", right_on="Time",
                         direction="nearest")

    out = laps.rename(columns={
        "Driver": "driver", "Team": "team", "LapNumber": "lap_number",
        "Compound": "compound", "TyreLife": "tyre_age", "Stint": "stint",
        "TrackTemp": "track_temp",
    })[COLUMNS]
    out = out.astype({"lap_number": int, "tyre_age": int, "stint": int})
    return out.sort_values(["driver", "lap_number"]).reset_index(drop=True)
