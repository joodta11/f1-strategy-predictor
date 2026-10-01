import dataclasses

import pytest

from src.model_api import RaceContext
from src.simulator import find_optimal_strategy, generate_strategies, simulate_strategy


@pytest.fixture
def ctx():
    return RaceContext(
        year=2025, gp_name="Test", total_laps=40, compounds=["SOFT", "MEDIUM", "HARD"],
        model_type="baseline",
        base_pace={"SOFT": 90.0, "MEDIUM": 90.4, "HARD": 90.8},
        deg_rate={"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04},
        deg_quadratic={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
        fuel_effect=-0.05, driver_offset=0.0,
        max_tyre_age={"SOFT": 40, "MEDIUM": 40, "HARD": 40}, pit_loss=22.0,
    )


def test_every_strategy_uses_two_compounds():
    strats = generate_strategies(40, ["SOFT", "MEDIUM", "HARD"], max_stops=2, min_stint=8)
    assert strats
    assert all(len({c for c, _ in s}) >= 2 for s in strats)
    assert [("SOFT", 0), ("SOFT", 20)] not in strats
    assert [("SOFT", 0), ("MEDIUM", 20)] in strats


def test_min_stint_and_max_stops():
    strats = generate_strategies(40, ["MEDIUM", "HARD"], max_stops=2, min_stint=8)
    for s in strats:
        assert len(s) <= 3
        edges = [p for _, p in s] + [40]
        assert min(b - a for a, b in zip(edges, edges[1:])) >= 8


def test_pit_loss_applied_per_stop(ctx):
    one = [("MEDIUM", 0), ("HARD", 20)]
    two = [("MEDIUM", 0), ("HARD", 15), ("HARD", 30)]
    for strat in (one, two):
        with_loss, _ = simulate_strategy(strat, ctx)
        no_loss, _ = simulate_strategy(strat, dataclasses.replace(ctx, pit_loss=0.0))
        assert with_loss - no_loss == pytest.approx(22.0 * (len(strat) - 1))


def test_per_lap_times_cover_race(ctx):
    total, laps = simulate_strategy([("SOFT", 0), ("HARD", 12)], ctx)
    assert len(laps) == 40
    assert sum(laps) == pytest.approx(total)


def test_ranking_sorted_and_matches_simulation(ctx):
    ranked = find_optimal_strategy(ctx, max_stops=2)
    assert ranked.total_time.is_monotonic_increasing
    assert ranked.gap_to_best.iloc[0] == 0
    for _, row in ranked.head(5).iterrows():
        assert simulate_strategy(row.strategy, ctx)[0] == pytest.approx(row.total_time)


def test_huge_pit_loss_prefers_one_stop(ctx):
    ranked = find_optimal_strategy(dataclasses.replace(ctx, pit_loss=200.0), max_stops=2)
    assert ranked.stops.iloc[0] == 1
