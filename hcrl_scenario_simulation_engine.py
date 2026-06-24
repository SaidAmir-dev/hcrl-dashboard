"""
HCRL Workforce Scenario Simulation Engine

Purpose:
Simulate how modeled attrition exposure could change under hypothetical
workforce improvement scenarios.

No causal claims.
No ROI claims.
No guaranteed savings.
No automatic employment decisions.
No firing recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class ScenarioSimulationReport:
    scenarios_generated: int
    baseline_exposure: float
    warnings: List[str]
    errors: List[str]


SCENARIO_TO_DRIVER_GROUP = {
    "Career Progression Improvement": "Career Progression",
    "Compensation Improvement": "Compensation",
    "Manager Stability Improvement": "Manager Stability",
    "Work Environment Improvement": "Work Environment",
    "Workload Reduction": "Workload",
    "Training and Development Improvement": "Training and Development",
    "Travel / Commute Burden Reduction": "Travel / Commute Burden",
}


def build_scenario_simulation_table(
    priority_table: pd.DataFrame,
    leverage_table: pd.DataFrame,
    scenario_name: str,
    scenario_intensity_pct: float,
) -> Tuple[pd.DataFrame, ScenarioSimulationReport]:

    warnings: List[str] = []
    errors: List[str] = []

    if "total_expected_attrition_cost" not in priority_table.columns:
        errors.append("priority_table requires total_expected_attrition_cost.")

    required_leverage_cols = [
        "driver_group",
        "leverage_score",
        "supporting_variables",
        "actionability",
    ]

    missing_leverage = [
        col for col in required_leverage_cols
        if col not in leverage_table.columns
    ]

    if missing_leverage:
        errors.append(f"leverage_table missing columns: {missing_leverage}")

    if scenario_name not in SCENARIO_TO_DRIVER_GROUP:
        errors.append(f"Unknown scenario name: {scenario_name}")

    if scenario_intensity_pct < 0:
        errors.append("scenario_intensity_pct cannot be negative.")

    if errors:
        return pd.DataFrame(), ScenarioSimulationReport(
            scenarios_generated=0,
            baseline_exposure=0.0,
            warnings=warnings,
            errors=errors,
        )

    baseline_exposure = float(
        pd.to_numeric(
            priority_table["total_expected_attrition_cost"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    if baseline_exposure <= 0:
        errors.append("Baseline exposure is zero or unavailable.")
        return pd.DataFrame(), ScenarioSimulationReport(
            scenarios_generated=0,
            baseline_exposure=0.0,
            warnings=warnings,
            errors=errors,
        )

    target_group = SCENARIO_TO_DRIVER_GROUP[scenario_name]

    leverage = leverage_table.copy()

    leverage["leverage_score"] = pd.to_numeric(
        leverage["leverage_score"],
        errors="coerce",
    )

    leverage = leverage[
        leverage["leverage_score"].notna()
    ].copy()

    matched = leverage[
        leverage["driver_group"] == target_group
    ].copy()

    if matched.empty:
        errors.append(
            f"No leverage evidence found for scenario driver group: {target_group}"
        )
        return pd.DataFrame(), ScenarioSimulationReport(
            scenarios_generated=0,
            baseline_exposure=baseline_exposure,
            warnings=warnings,
            errors=errors,
        )

    evidence_row = matched.iloc[0]

    leverage_score = float(evidence_row["leverage_score"])

    scenario_intensity = scenario_intensity_pct / 100.0

    modeled_exposure_change_pct = leverage_score * scenario_intensity

    scenario_exposure = baseline_exposure * (
        1 - modeled_exposure_change_pct
    )

    exposure_difference = baseline_exposure - scenario_exposure

    output = pd.DataFrame(
        [
            {
                "scenario_name": scenario_name,
                "driver_group": target_group,
                "scenario_intensity_pct": scenario_intensity_pct,
                "leverage_score": leverage_score,
                "modeled_exposure_change_pct": modeled_exposure_change_pct,
                "baseline_exposure": baseline_exposure,
                "scenario_exposure": scenario_exposure,
                "exposure_difference": exposure_difference,
                "supporting_variables": evidence_row.get(
                    "supporting_variables",
                    "",
                ),
                "actionability": evidence_row.get(
                    "actionability",
                    "",
                ),
                "scenario_summary": (
                    f"{scenario_name} is modeled using the observed leverage score "
                    f"for {target_group}. Under a {scenario_intensity_pct:.0f}% "
                    f"scenario intensity, modeled exposure changes by approximately "
                    f"{modeled_exposure_change_pct:.2%}."
                ),
            }
        ]
    )

    warnings.append(
        "Scenario simulation is based on observed leverage evidence and user-selected "
        "scenario intensity. It is not causal forecasting, guaranteed savings, ROI, "
        "or an automatic employment decision."
    )

    return output, ScenarioSimulationReport(
        scenarios_generated=len(output),
        baseline_exposure=baseline_exposure,
        warnings=warnings,
        errors=errors,
    )
