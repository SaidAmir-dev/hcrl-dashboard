"""
HCRL Workforce Leverage Intelligence Engine

Purpose:
Aggregate cleaned attrition drivers into executive-level workforce themes.

This engine helps answer:
"Which workforce domains show the strongest combined evidence of association with attrition risk?"

No causal claims.
No automated decisions.
No intervention prescriptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class WorkforceLeverageReport:
    leverage_areas_identified: int
    warnings: List[str]
    errors: List[str]


VALID_LEVERAGE_GROUPS = {
    "Compensation",
    "Career Progression",
    "Employee Experience",
    "Manager Stability",
    "Workload",
    "Work Environment",
    "Travel / Commute Burden",
    "Training and Development",
    "Department",
    "Occupation",
    "Education",
    "Performance",
}


def build_workforce_leverage_table(
    driver_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, WorkforceLeverageReport]:

    warnings: List[str] = []
    errors: List[str] = []

    required_cols = [
        "driver_group",
        "driver_variable",
        "association_value",
        "actionability",
    ]

    missing = [
        col for col in required_cols
        if col not in driver_table.columns
    ]

    if missing:
        errors.append(f"Missing required columns: {missing}")
        return pd.DataFrame(), WorkforceLeverageReport(0, warnings, errors)

    df = driver_table.copy()

    df = df[df["driver_group"].isin(VALID_LEVERAGE_GROUPS)].copy()

    df["association_value"] = pd.to_numeric(
        df["association_value"],
        errors="coerce",
    )

    df = df[df["association_value"].notna()].copy()

    if df.empty:
        errors.append("No valid cleaned driver evidence available.")
        return pd.DataFrame(), WorkforceLeverageReport(0, warnings, errors)

    df["absolute_association"] = df["association_value"].abs()

    leverage = (
        df.groupby("driver_group")
        .agg(
            evidence_drivers=("driver_variable", "nunique"),
            supporting_variables=(
                "driver_variable",
                lambda x: " | ".join(sorted(set(map(str, x)))),
            ),
            strongest_association=("absolute_association", "max"),
            average_association=("absolute_association", "mean"),
            actionability=("actionability", "first"),
        )
        .reset_index()
    )

    leverage["leverage_score"] = (
        leverage["average_association"]
        * leverage["evidence_drivers"]
    )

    leverage = leverage.sort_values(
        ["actionability", "leverage_score", "strongest_association"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    leverage["leverage_rank"] = range(1, len(leverage) + 1)

    leverage = leverage[
        [
            "leverage_rank",
            "driver_group",
            "evidence_drivers",
            "supporting_variables",
            "strongest_association",
            "average_association",
            "leverage_score",
            "actionability",
        ]
    ]

    warnings.append(
        "Leverage areas are aggregated from cleaned attrition driver evidence. "
        "They identify management domains statistically associated with attrition risk, "
        "but they do not establish causality or prescribe interventions."
    )

    return leverage, WorkforceLeverageReport(
        leverage_areas_identified=len(leverage),
        warnings=warnings,
        errors=errors,
    )
