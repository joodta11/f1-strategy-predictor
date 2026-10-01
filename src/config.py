"""Races used in the project: fully dry 2025 races at different degradation levels.

Each was checked as dry: only SOFT/MEDIUM/HARD compounds used and the weather
feed never reported rainfall.
"""

YEAR = 2025

RACES = [
    # (year, FastF1 event name, degradation character)
    (2025, "Bahrain", "high"),
    (2025, "Hungary", "medium"),
    (2025, "Japan", "low"),
]

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
