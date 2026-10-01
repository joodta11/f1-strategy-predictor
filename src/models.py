"""Tyre degradation models: a physics-style linear baseline and a gradient boosted model.

Baseline (fitted per race):
    lap_time = base_pace[compound] + driver_offset + deg_rate[compound] * tyre_age
               (+ deg_quadratic[compound] * tyre_age^2) + fuel_effect * lap_number

Compound-specific terms give a separate degradation rate per compound, while the
fuel effect and driver offsets are shared across compounds. Fitting compounds
separately would make the fuel effect unidentifiable (e.g. in Japan the soft was
only run at the end of the race, so tyre age and lap number move together).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


# ---------------------------------------------------------------- baseline ---

def _design(laps: pd.DataFrame, drivers, compounds, quadratic: bool) -> pd.DataFrame:
    X = {}
    for c in compounds:
        is_c = (laps.compound == c).astype(float)
        X[f"is_{c}"] = is_c
        X[f"age_{c}"] = is_c * laps.tyre_age
        if quadratic:
            X[f"age2_{c}"] = is_c * laps.tyre_age ** 2
    X["lap_number"] = laps.lap_number.astype(float)
    for d in drivers[1:]:  # first driver is the reference level
        X[f"drv_{d}"] = (laps.driver == d).astype(float)
    return pd.DataFrame(X, index=laps.index)


@dataclass
class BaselineModel:
    quadratic: bool = False
    drivers: list = field(default_factory=list)
    compounds: list = field(default_factory=list)
    reg: LinearRegression = None

    def fit(self, laps: pd.DataFrame) -> "BaselineModel":
        self.drivers = sorted(laps.driver.unique())
        self.compounds = [c for c in COMPOUNDS if c in set(laps.compound)]
        X = _design(laps, self.drivers, self.compounds, self.quadratic)
        self.reg = LinearRegression(fit_intercept=False).fit(X, laps.lap_time_s)
        return self

    def predict(self, laps: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(_design(laps, self.drivers, self.compounds, self.quadratic))

    @property
    def coef(self) -> dict:
        return dict(zip(self.reg.feature_names_in_, self.reg.coef_))

    def deg_rate(self, compound) -> float:
        return self.coef[f"age_{compound}"]

    def deg_quadratic(self, compound) -> float:
        return self.coef.get(f"age2_{compound}", 0.0)

    def base_pace(self, compound) -> float:
        return self.coef[f"is_{compound}"]

    @property
    def fuel_effect(self) -> float:
        return self.coef["lap_number"]

    def driver_offsets(self) -> pd.Series:
        offs = {self.drivers[0]: 0.0}
        offs.update({d: self.coef[f"drv_{d}"] for d in self.drivers[1:]})
        return pd.Series(offs)


# ------------------------------------------------------------ ML model -------

ML_FEATURES = ["compound", "tyre_age", "lap_number", "track_temp", "driver", "race"]
ML_CATEGORICAL = ["compound", "driver", "race"]


def _ml_X(laps: pd.DataFrame) -> pd.DataFrame:
    X = laps[ML_FEATURES].copy()
    for c in ML_CATEGORICAL:
        X[c] = X[c].astype(str).astype("category")
    return X


def make_gbm() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(categorical_features="from_dtype", max_iter=300,
                                         learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=20, random_state=0)


class GBMModel:
    def fit(self, laps: pd.DataFrame, target: str = "lap_time_s") -> "GBMModel":
        X = _ml_X(laps)
        self.categories = {c: X[c].cat.categories for c in ML_CATEGORICAL}
        self.reg = make_gbm().fit(X, laps[target])
        return self

    def predict(self, laps: pd.DataFrame) -> np.ndarray:
        X = _ml_X(laps)
        for c, cats in self.categories.items():  # unseen categories become missing
            X[c] = pd.Categorical(X[c].astype(str), categories=cats)
        return self.reg.predict(X)


# -------------------------------------------------------------- evaluation ---

def stint_split(laps: pd.DataFrame, test_size=0.2, seed=0):
    """Hold out whole stints so laps from the same stint never sit on both sides."""
    groups = laps.race.astype(str) + "|" + laps.driver + "|" + laps.stint.astype(str)
    train_idx, test_idx = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed)
                               .split(laps, groups=groups))
    return laps.iloc[train_idx], laps.iloc[test_idx]


def mae_by_compound(test: pd.DataFrame, pred) -> dict:
    err = (test.lap_time_s - pred).abs()
    out = err.groupby(test.compound).mean().to_dict()
    out["ALL"] = mean_absolute_error(test.lap_time_s, pred)
    return out


def add_relative_target(laps: pd.DataFrame) -> pd.DataFrame:
    """Lap time relative to the driver's median in that race, comparable across tracks."""
    med = laps.groupby(["race", "driver"]).lap_time_s.transform("median")
    return laps.assign(rel_time=laps.lap_time_s - med)


def _pooled_baseline_X(laps):
    X = {}
    for c in COMPOUNDS:
        is_c = (laps.compound == c).astype(float)
        X[f"is_{c}"] = is_c
        X[f"age_{c}"] = is_c * laps.tyre_age
    X["lap_frac"] = laps.lap_number / laps.total_laps
    return pd.DataFrame(X, index=laps.index)


def leave_one_race_out(all_laps: pd.DataFrame) -> pd.DataFrame:
    """Train on two races, test on the third.

    Absolute pace on an unseen track can't be learned from other tracks, so both
    models predict lap time relative to each driver's race median. The baseline
    becomes a pooled linear model (compound offsets, per-compound deg rate,
    fuel effect per fraction of race distance).
    """
    laps = add_relative_target(all_laps)
    rows = []
    for race in laps.race.unique():
        train, test = laps[laps.race != race], laps[laps.race == race]
        lin = LinearRegression().fit(_pooled_baseline_X(train), train.rel_time)
        gbm = GBMModel().fit(train, target="rel_time")
        for name, pred in [("baseline", lin.predict(_pooled_baseline_X(test))),
                           ("gbm", gbm.predict(test))]:
            rows.append({"race": race, "model": name,
                         "mae": mean_absolute_error(test.rel_time, pred)})
    return pd.DataFrame(rows)
