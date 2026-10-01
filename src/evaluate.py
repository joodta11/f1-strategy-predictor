"""Fit and compare the degradation models on the configured races.

Run from the project root:  python -m src.evaluate
"""
import logging

import pandas as pd

from src.cleaning import clean_laps
from src.config import RACES
from src.data_loader import load_race
from src.models import (BaselineModel, GBMModel, leave_one_race_out, mae_by_compound,
                        stint_split)


def load_all_clean() -> pd.DataFrame:
    frames = []
    for year, gp, _ in RACES:
        s = load_race(year, gp)
        frames.append(clean_laps(s).assign(race=gp, total_laps=int(s.laps.LapNumber.max())))
    return pd.concat(frames, ignore_index=True)


def main():
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    laps = load_all_clean()
    train, test = stint_split(laps)
    rows, coef_rows = [], []

    for race in laps.race.unique():
        tr, te = train[train.race == race], test[test.race == race]
        for quad in (False, True):
            m = BaselineModel(quadratic=quad).fit(tr)
            name = "baseline_quad" if quad else "baseline"
            for comp, mae in mae_by_compound(te, m.predict(te)).items():
                rows.append({"race": race, "model": name, "compound": comp, "mae": mae})
            if not quad:
                for c in m.compounds:
                    coef_rows.append({"race": race, "compound": c,
                                      "deg_rate_s_per_lap": m.deg_rate(c),
                                      "fuel_effect_s_per_lap": m.fuel_effect})

    gbm = GBMModel().fit(train)
    for race in laps.race.unique():
        te = test[test.race == race]
        for comp, mae in mae_by_compound(te, gbm.predict(te)).items():
            rows.append({"race": race, "model": "gbm", "compound": comp, "mae": mae})

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print("\nHeld-out stints (20%), MAE in seconds:")
    print(res.pivot_table(index=["race", "compound"], columns="model", values="mae").round(3))
    print("\nFitted linear baseline coefficients:")
    print(pd.DataFrame(coef_rows).round(3).to_string(index=False))

    loro = leave_one_race_out(laps)
    print("\nLeave-one-race-out, MAE of lap time relative to the driver's race median (s):")
    print(loro.pivot(index="race", columns="model", values="mae").round(3))
    print("\nGBM train-set MAE (overfitting check):",
          round(mae_by_compound(train, gbm.predict(train))["ALL"], 3))


if __name__ == "__main__":
    main()
