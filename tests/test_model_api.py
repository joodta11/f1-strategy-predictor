import numpy as np
import pytest

from src.model_api import RaceContext, predict_lap_time, predict_lap_times


@pytest.fixture
def ctx():
    return RaceContext(
        year=2025, gp_name="Test", total_laps=50, compounds=["SOFT", "MEDIUM", "HARD"],
        model_type="baseline",
        base_pace={"SOFT": 90.0, "MEDIUM": 90.4, "HARD": 90.8},
        deg_rate={"SOFT": 0.10, "MEDIUM": 0.06, "HARD": 0.03},
        deg_quadratic={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
        fuel_effect=-0.05, driver_offset=0.3,
        max_tyre_age={"SOFT": 20, "MEDIUM": 30, "HARD": 40}, pit_loss=22.0,
    )


@pytest.mark.parametrize("compound", ["SOFT", "MEDIUM", "HARD"])
def test_lap_time_rises_with_tyre_age(ctx, compound):
    times = [predict_lap_time(compound, age, 20, ctx) for age in (1, 10, 20)]
    assert times[0] < times[1] < times[2]


@pytest.mark.parametrize("compound", ["SOFT", "MEDIUM", "HARD"])
def test_lap_time_falls_with_lap_number(ctx, compound):
    times = [predict_lap_time(compound, 5, lap, ctx) for lap in (5, 25, 45)]
    assert times[0] > times[1] > times[2]


def test_vectorised_matches_scalar(ctx):
    ages, laps = np.arange(1, 11), np.arange(11, 21)
    vec = predict_lap_times("MEDIUM", ages, laps, ctx)
    assert np.allclose(vec, [predict_lap_time("MEDIUM", a, l, ctx) for a, l in zip(ages, laps)])


@pytest.mark.slow
def test_fitted_real_race_context_behaves():
    """Uses the FastF1 cache (downloads on first run)."""
    from src.model_api import fit_race_context

    real = fit_race_context(2025, "Bahrain")
    for c in real.compounds:
        assert predict_lap_time(c, 20, 30, real) > predict_lap_time(c, 2, 30, real)
        assert predict_lap_time(c, 10, 50, real) < predict_lap_time(c, 10, 10, real)
