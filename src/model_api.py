"""A small prediction interface on top of the fitted degradation model.

    ctx = fit_race_context(2025, "Bahrain")
    predict_lap_time("MEDIUM", tyre_age=10, lap_number=25, race_context=ctx)
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.cleaning import clean_laps
from src.data_loader import load_race
from src.models import BaselineModel, GBMModel

DEFAULT_MODEL = "baseline"  # beat the GBM on held-out stints and unseen races (src/evaluate.py)


@dataclass
class RaceContext:
    year: int
    gp_name: str
    total_laps: int
    compounds: list
    model_type: str
    base_pace: dict            # per compound, for the reference driver at lap 0, tyre age 0
    deg_rate: dict             # seconds lost per lap of tyre age, per compound
    deg_quadratic: dict
    fuel_effect: float         # seconds per lap (negative: the car gets lighter)
    driver_offset: float       # median driver offset, i.e. a representative driver
    max_tyre_age: dict         # longest stint observed per compound; the model is not trusted beyond it
    pit_loss: float = float("nan")
    track_temp: float = float("nan")
    representative_driver: str = ""
    gbm: object = field(default=None, repr=False)


def fit_race_context(year: int, gp_name: str, model: str = DEFAULT_MODEL,
                     session=None) -> RaceContext:
    from src.pit_loss import estimate_pit_loss

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    session = session if session is not None else load_race(year, gp_name)
    laps = clean_laps(session).assign(race=gp_name)

    base = BaselineModel().fit(laps)
    offsets = base.driver_offsets()
    median_offset = float(offsets.median())
    rep_driver = (offsets - median_offset).abs().idxmin()

    raw = session.laps
    ctx = RaceContext(
        year=year, gp_name=gp_name,
        total_laps=int(raw.LapNumber.max()),
        compounds=list(base.compounds),
        model_type=model,
        base_pace={c: base.base_pace(c) for c in base.compounds},
        deg_rate={c: base.deg_rate(c) for c in base.compounds},
        deg_quadratic={c: base.deg_quadratic(c) for c in base.compounds},
        fuel_effect=base.fuel_effect,
        driver_offset=median_offset,
        max_tyre_age=raw[raw.Compound.isin(base.compounds)]
            .groupby("Compound").TyreLife.max().astype(int).to_dict(),
        pit_loss=estimate_pit_loss(session),
        track_temp=float(laps.track_temp.median()),
        representative_driver=rep_driver,
    )
    if model == "gbm":
        ctx.gbm = GBMModel().fit(laps)
    elif model != "baseline":
        raise ValueError(f"unknown model {model!r}")
    return ctx


def predict_lap_times(compound: str, tyre_age, lap_number, race_context: RaceContext) -> np.ndarray:
    """Vectorised lap-time prediction for a representative (median) driver."""
    age = np.asarray(tyre_age, dtype=float)
    lap = np.asarray(lap_number, dtype=float)
    ctx = race_context
    if ctx.model_type == "gbm":
        n = np.broadcast(age, lap).size
        X = pd.DataFrame({"compound": [compound] * n, "tyre_age": np.broadcast_to(age, n),
                          "lap_number": np.broadcast_to(lap, n),
                          "track_temp": ctx.track_temp,
                          "driver": ctx.representative_driver, "race": ctx.gp_name})
        return ctx.gbm.predict(X)
    return (ctx.base_pace[compound] + ctx.driver_offset
            + ctx.deg_rate[compound] * age + ctx.deg_quadratic[compound] * age ** 2
            + ctx.fuel_effect * lap)


def predict_lap_time(compound: str, tyre_age: float, lap_number: float,
                     race_context: RaceContext) -> float:
    return float(predict_lap_times(compound, [tyre_age], [lap_number], race_context)[0])
