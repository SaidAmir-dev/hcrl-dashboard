"""
HCRL Workforce Leverage Intelligence Engine

Purpose:
Aggregate attrition drivers into executive-level workforce themes.

This engine helps answer:

"Which workforce domains show the strongest evidence
of association with attrition risk?"

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


def build_workforce_leverage_table(
    driver_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, WorkforceLeverageReport]:

    warnings: List[str] = []
    errors: List[str] = []

    required_cols = [
        "driver_group",
        "association_value",
        "actionability",
    ]

    missing = [
        col for col in required_cols
        if col not in driver_table.columns
    ]

    if missing:
        errors.append(
            f"Missing required columns: {missing}"
        )

        return (
            pd.DataFrame(),
            WorkforceLeverageReport(
                leverage_areas_identified=0,
                warnings=warnings,
                errors=errors,
            ),
        )

    df = driver_table.copy()

    df["association_value"] = pd.to_numeric(
        df["association_value"],
        errors="coerce",
    )

    df = df[
        df["association_value"].notna()
    ].copy()

    if df.empty:
        errors.append(
            "No valid driver evidence available."
        )

        return (
            pd.DataFrame(),
            WorkforceLeverageReport(
                leverage_areas_identified=0,
                warnings=warnings,
                errors=errors,
            ),
        )

    df["absolute_association"] = (
        df["association_value"].abs()
    )

    leverage = (
        df.groupby("driver_group")
        .agg(
            evidence_drivers=(
                "driver_group",
                "count",
            ),
            strongest_association=(
                "absolute_association",
                "max",
            ),
            average_association=(
                "absolute_association",
                "mean",
            ),
            actionability=(
                "actionability",
                "first",
            ),
        )
        .reset_index()
    )

    leverage = leverage.sort_values(
        "strongest_association",
        ascending=False,
    )

    leverage["leverage_rank"] = range(
        1,
        len(leverage) + 1,
    )

    leverage = leverage[
        [
            "leverage_rank",
            "driver_group",
            "evidence_drivers",
            "strongest_association",
            "average_association",
            "actionability",
        ]
    ]

    warnings.append(
        "Leverage areas represent aggregated evidence from the attrition driver engine. "
        "They identify areas associated with attrition risk but do not establish causality."
    )

    return (
        leverage,
        WorkforceLeverageReport(
            leverage_areas_identified=len(leverage),
            warnings=warnings,
            errors=errors,
        ),
    )
