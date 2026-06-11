"""
HCRL Workforce Action Intelligence Engine

Purpose:
Convert workforce analytics into executive actions.

No firing recommendations.
No arbitrary thresholds.
No hardcoded risk bands.

Uses:
- attrition risk
- economic exposure
- work profile evidence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class ActionIntelligenceReport:
    occupations_analyzed: int
    warnings: List[str]
    errors: List[str]


def build_action_intelligence_table(
    prioritization_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, ActionIntelligenceReport]:

    warnings = []
    errors = []

    required_cols = [
        "matched_onet_title",
        "share_of_total_cost_pct",
        "avg_attrition_probability",
        "primary_work_type",
    ]

    missing = [
        c for c in required_cols
        if c not in prioritization_df.columns
    ]

    if missing:
        errors.append(
            f"Missing required columns: {missing}"
        )

        return (
            pd.DataFrame(),
            ActionIntelligenceReport(
                occupations_analyzed=0,
                warnings=warnings,
                errors=errors,
            ),
        )

    df = prioritization_df.copy()

    df["cost_percentile"] = (
        df["share_of_total_cost_pct"]
        .rank(pct=True)
    )

    df["risk_percentile"] = (
        df["avg_attrition_probability"]
        .rank(pct=True)
    )

    def determine_action(row):

        primary = str(
            row["primary_work_type"]
        )

        cost_pct = row["cost_percentile"]
        risk_pct = row["risk_percentile"]

        if (
            cost_pct >= 0.75
            and risk_pct >= 0.75
        ):
            return "Retention Priority"

        if (
            primary == "Digital"
            and cost_pct >= 0.50
        ):
            return "Automation Review"

        if (
            primary == "Analytical"
        ):
            return "AI Augmentation"

        if (
            primary == "Human Interaction"
        ):
            return "Human-Centered Workforce"

        if (
            primary == "Physical / Manual"
        ):
            return "Process Improvement"

        return "Further Review"

    df["recommended_action"] = (
        df.apply(
            determine_action,
            axis=1,
        )
    )

    def build_reason(row):

        return (
            f"Cost Share={row['share_of_total_cost_pct']:.1f}% | "
            f"Attrition Risk={row['avg_attrition_probability']:.1%} | "
            f"Primary Work={row['primary_work_type']}"
        )

    df["action_rationale"] = (
        df.apply(
            build_reason,
            axis=1,
        )
    )

    final = df[
        [
            "priority_rank",
            "matched_onet_title",
            "share_of_total_cost_pct",
            "avg_attrition_probability",
            "primary_work_type",
            "recommended_action",
            "action_rationale",
        ]
    ]

    return (
        final,
        ActionIntelligenceReport(
            occupations_analyzed=len(final),
            warnings=warnings,
            errors=errors,
        ),
    )
