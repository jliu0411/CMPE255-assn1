"""Transparent, salary-cap-aware NBA fair-value calculations.

The trained ML model estimates the historical contract benchmark. Fair value
is a separate decision-support estimate because contract salary is affected by
when a deal was signed. This module combines peer performance percentiles with
NBA maximum-salary eligibility and reports every assumption.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


# USD millions. Years correspond to the season-ending year used by the dataset.
# 2027+ values use the CBA's 10% maximum annual-growth planning assumption.
SALARY_CAP_MILLIONS = {
    2010: 57.70, 2011: 58.04, 2012: 58.04, 2013: 58.68,
    2014: 58.68, 2015: 63.07, 2016: 70.00, 2017: 94.14,
    2018: 99.09, 2019: 101.87, 2020: 109.14, 2021: 109.14,
    2022: 112.41, 2023: 123.66, 2024: 136.02, 2025: 140.59,
    2026: 154.65,
}

PERFORMANCE_FEATURES = {
    "PTS": ("Points_Per_Game", 0.35),
    "AST": ("Assists_Per_Game", 0.18),
    "REB": ("Rebounds_Per_Game", 0.16),
    "STL": ("Steals_Per_Game", 0.10),
    "BLK": ("Blocks_Per_Game", 0.08),
    "MIN": ("Minutes_Per_Game", 0.13),
}


def salary_cap_for_year(year: int) -> tuple[float, bool]:
    """Return the cap and whether it is projected rather than historical."""
    if year in SALARY_CAP_MILLIONS:
        return SALARY_CAP_MILLIONS[year], False
    if year < min(SALARY_CAP_MILLIONS):
        raise ValueError(f"Valuation year must be {min(SALARY_CAP_MILLIONS)} or later.")
    latest_year = max(SALARY_CAP_MILLIONS)
    return SALARY_CAP_MILLIONS[latest_year] * (1.10 ** (year - latest_year)), True


def _percentile(value: float, reference: pd.Series) -> float:
    clean = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return 0.5
    return float(np.mean(clean <= value))


def performance_percentile(payload: Mapping[str, object], reference: pd.DataFrame) -> float:
    """Weighted empirical percentile among player seasons in the source data."""
    score = 0.0
    for input_name, (column, weight) in PERFORMANCE_FEATURES.items():
        score += weight * _percentile(float(payload[input_name]), reference[column])
    return float(np.clip(score, 0.0, 1.0))


def maximum_salary_rate(years_of_service: float, mvp: bool, all_nba: bool) -> float:
    """Simplified CBA max tier, including designated-veteran eligibility."""
    if years_of_service >= 10:
        return 0.35
    if years_of_service >= 7:
        return 0.35 if (mvp or all_nba) else 0.30
    return 0.25


def estimate_fair_value(
    payload: Mapping[str, object],
    contract_benchmark_millions: float,
    reference: pd.DataFrame,
) -> dict[str, float | int | bool | str]:
    stats_year = int(float(payload.get("Stats_Year", payload.get("Valuation_Year", 2025))))
    contract_year = int(float(payload.get("Contract_Start_Year", payload.get("Valuation_Year", stats_year))))
    if not 2010 <= stats_year <= 2025:
        raise ValueError("Stats season must be between 2010 and 2025, matching the training data.")
    if contract_year < stats_year:
        raise ValueError("Contract start year cannot be earlier than the stats season.")
    if contract_year > 2035:
        raise ValueError("Contract start year must be 2035 or earlier.")

    stats_years_of_service = float(payload.get("Years_in_League", 5))
    contract_years_of_service = stats_years_of_service + (contract_year - stats_year)
    mvp = str(payload.get("MVP", "No")).lower() == "yes"
    all_nba = str(payload.get("All_NBA", "No")).lower() == "yes"
    all_star = str(payload.get("All_Star", "No")).lower() == "yes"

    cap, projected = salary_cap_for_year(contract_year)
    percentile = performance_percentile(payload, reference)
    max_rate = maximum_salary_rate(contract_years_of_service, mvp, all_nba)
    first_year_max = cap * max_rate

    # Translate empirical performance rank into a share of the eligible max.
    # The nonlinear curve reserves max-level values for genuinely elite peers.
    performance_strength = float(np.clip((percentile - 0.35) / 0.63, 0.0, 1.0)) ** 1.5
    performance_value = first_year_max * (0.08 + 0.92 * performance_strength)
    if all_star:
        performance_value = max(performance_value, first_year_max * 0.65)
    if all_nba:
        performance_value = max(performance_value, first_year_max * 0.88)
    if mvp:
        performance_value = first_year_max

    fair_value = min(first_year_max, max(contract_benchmark_millions, performance_value))
    four_year_total = fair_value * sum(1 + 0.08 * i for i in range(4))

    return {
        "stats_year": stats_year,
        "contract_start_year": contract_year,
        "contract_years_of_service": round(contract_years_of_service, 1),
        "salary_cap_millions": round(cap, 3),
        "cap_is_projected": projected,
        "performance_percentile": round(percentile * 100, 1),
        "max_salary_rate": round(max_rate * 100, 1),
        "eligible_first_year_max_millions": round(first_year_max, 2),
        "contract_benchmark_millions": round(contract_benchmark_millions, 2),
        "fair_value_millions": round(fair_value, 2),
        "four_year_total_millions": round(four_year_total, 2),
        "four_year_aav_millions": round(four_year_total / 4, 2),
        "honors_adjustment": "MVP" if mvp else "All-NBA" if all_nba else "All-Star" if all_star else "None",
    }
