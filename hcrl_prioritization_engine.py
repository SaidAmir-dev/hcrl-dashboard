"""HCRL Workforce Prioritization Engine.

Purpose:
Identify which occupations create the largest workforce risk and
economic exposure.

No arbitrary thresholds.
No hardcoded weights.
No automated employment decisions.

This module ranks occupations using observed outputs:
- attrition probability
- expected attrition cost
- workforce size
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class PrioritizationReport:
    occupations_analyzed: int
    warnings: List[str]
    errors: List[str]


def build_prioritization_table(
    workforce_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, PrioritizationReport]:

    warnings = []
    errors = []

    required_cols = [
        "matched_onet_title",
        "predicted_attrition_probability",
        "expected_attrition_cost",
    ]

    missing = [c for c in required_cols if c not in workforce_df.columns]

    if missing:
        errors.append(
            f"Missing required columns: {missing}"
        )

        return pd.DataFrame(), PrioritizationReport(
            occupations_analyzed=0,
            warnings=warnings,
            errors=errors,
        )

    df = workforce_df.copy()

    df = df[
        df["matched_onet_title"].notna()
    ].copy()

    if len(df) == 0:
        errors.append("No mapped occupations available.")

        return pd.DataFrame(), PrioritizationReport(
            occupations_analyzed=0,
            warnings=warnings,
            errors=errors,
        )

    summary = (
        df.groupby("matched_onet_title")
        .agg(
            n_workers=("matched_onet_title", "count"),
            avg_attrition_probability=(
                "predicted_attrition_probability",
                "mean",
            ),
            total_expected_attrition_cost=(
                "expected_attrition_cost",
                "sum",
            ),
            avg_digital_work=(
                "ai_digital_work_percentile",
                "mean",
            ),
            avg_analytical_work=(
                "ai_analytical_cognitive_work_percentile",
                "mean",
            ),
            avg_human_interaction=(
                "ai_human_interaction_work_percentile",
                "mean",
            ),
            avg_physical_work=(
                "ai_physical_manual_work_percentile",
                "mean",
            ),
        )
        .reset_index()
    )

    total_cost = summary[
        "total_expected_attrition_cost"
    ].sum()

    summary["share_of_total_cost"] = (
        summary["total_expected_attrition_cost"]
        / total_cost
    )

    summary["cost_rank"] = (
        summary["total_expected_attrition_cost"]
        .rank(
            ascending=False,
            method="dense",
        )
    )

    summary["risk_rank"] = (
        summary["avg_attrition_probability"]
        .rank(
            ascending=False,
            method="dense",
        )
    )

    summary["worker_rank"] = (
        summary["n_workers"]
        .rank(
            ascending=False,
            method="dense",
        )
    )

    summary["priority_score"] = (
        summary["cost_rank"]
        + summary["risk_rank"]
        + summary["worker_rank"]
    )

    summary = summary.sort_values(
        "priority_score"
    )

    summary["priority_rank"] = range(
        1,
        len(summary) + 1,
    )

    summary = summary[
        [
            "priority_rank",
            "matched_onet_title",
            "n_workers",
            "avg_attrition_probability",
            "total_expected_attrition_cost",
            "share_of_total_cost",
            "avg_digital_work",
            "avg_analytical_work",
            "avg_human_interaction",
            "avg_physical_work",
        ]
    ]

    return summary, PrioritizationReport(
        occupations_analyzed=len(summary),
        warnings=warnings,
        errors=errors,
    )
