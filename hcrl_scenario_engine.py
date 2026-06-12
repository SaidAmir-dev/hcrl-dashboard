"""
HCRL Workforce Scenario Engine

Purpose:
Estimate potential economic savings under transparent attrition-reduction
scenarios.

No predictions.
No arbitrary risk thresholds.
No automated employment decisions.

Scenarios are user-facing assumptions:
- What if expected attrition cost is reduced by 10%?
- 20%?
- 30%?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class ScenarioReport:
    occupations_analyzed: int
    portfolio_current_cost: float
    warnings: List[str]
    errors: List[str]


def build_scenario_table(
    prioritization_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, ScenarioReport]:

    warnings: List[str] = []
    errors: List[str] = []

    required_cols = [
        "matched_onet_title",
        "total_expected_attrition_cost",
    ]

    missing = [
        col for col in required_cols
        if col not in prioritization_df.columns
    ]

    if missing:
        errors.append(f"Missing required columns: {missing}")
        return pd.DataFrame(), ScenarioReport(
            occupations_analyzed=0,
            portfolio_current_cost=0.0,
            warnings=warnings,
            errors=errors,
        )

    df = prioritization_df.copy()

    df["total_expected_attrition_cost"] = pd.to_numeric(
        df["total_expected_attrition_cost"],
        errors="coerce",
    ).fillna(0)

    portfolio_current_cost = float(
        df["total_expected_attrition_cost"].sum()
    )

    scenario_table = df[
        [
            "matched_onet_title",
            "n_workers",
            "total_expected_attrition_cost",
            "share_of_total_cost_pct",
        ]
    ].copy()

    scenario_table["savings_if_attrition_cost_reduced_10pct"] = (
        scenario_table["total_expected_attrition_cost"] * 0.10
    )

    scenario_table["savings_if_attrition_cost_reduced_20pct"] = (
        scenario_table["total_expected_attrition_cost"] * 0.20
    )

    scenario_table["savings_if_attrition_cost_reduced_30pct"] = (
        scenario_table["total_expected_attrition_cost"] * 0.30
    )

    scenario_table = scenario_table.sort_values(
        "total_expected_attrition_cost",
        ascending=False,
    )

    return scenario_table, ScenarioReport(
        occupations_analyzed=len(scenario_table),
        portfolio_current_cost=portfolio_current_cost,
        warnings=warnings,
        errors=errors,
    )
