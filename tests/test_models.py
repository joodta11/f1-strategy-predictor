import numpy as np
import pandas as pd

from src.models import BaselineModel, stint_split


def _synthetic_laps(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    deg = {"SOFT": 0.10, "MEDIUM": 0.06, "HARD": 0.03}
    base = {"SOFT": 90.0, "MEDIUM": 90.4, "HARD": 90.8}
    for i, drv in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        plan = [("SOFT", 1, 15), ("MEDIUM", 16, 35), ("HARD", 36, 55)][::(1 if i % 2 else -1)]
        start = 1
        for stint, (comp, _, _) in enumerate(plan, 1):
            for age in range(1, 19):
                lap = start + age - 1
                t = base[comp] + 0.2 * i + deg[comp] * age - 0.05 * lap + rng.normal(0, 0.05)
                rows.append({"driver": drv, "lap_number": lap, "lap_time_s": t,
                             "compound": comp, "tyre_age": age, "stint": stint,
                             "race": "Test"})
            start += 18
    return pd.DataFrame(rows)


def test_baseline_recovers_known_coefficients():
    m = BaselineModel().fit(_synthetic_laps())
    assert abs(m.deg_rate("SOFT") - 0.10) < 0.01
    assert abs(m.deg_rate("HARD") - 0.03) < 0.01
    assert abs(m.fuel_effect + 0.05) < 0.005


def test_stint_split_keeps_stints_whole():
    laps = _synthetic_laps()
    train, test = stint_split(laps)
    key = lambda df: set(zip(df.driver, df.stint))
    assert key(train).isdisjoint(key(test))
