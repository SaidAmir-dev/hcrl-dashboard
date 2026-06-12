"""
HCRL Executive Report Engine

Purpose:
Convert workforce analytics into an executive-ready report.

No automated employment decisions.
No firing recommendations.
No arbitrary risk bands.

The report identifies:
- largest workforce cost drivers
- largest attrition exposures
- management focus areas
- scenario opportunities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class ExecutiveReport:
    total_expected_attrition_cost: float
    top_cost_driver: str
    top_cost_share: float
    focus_areas_identified: int
    warnings: List[str]
    errors: List[str]


def build_executive_focus_table(
    priority_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, ExecutiveReport]:

    warnings = []
    errors = []

    required_cols = [
        "matched_onet_title",
        "share_of_total_cost_pct",
        "total_expected_attrition_cost",
        "primary_work_type",
    ]

    missing = [
        c for c in required_cols
        if c not in priority_table.columns
    ]

    if missing:
        errors.append(
            f"Missing required columns: {missing}"
        )

        return (
            pd.DataFrame(),
            ExecutiveReport(
                0,
                "",
                0,
                0,
                warnings,
                errors,
            ),
        )

    df = priority_table.copy()

    total_cost = float(
        df["total_expected_attrition_cost"].sum()
    )

    df = df.sort_values(
        "total_expected_attrition_cost",
        ascending=False,
    )

    def build_action(row):

        primary = str(
            row["primary_work_type"]
        )

        if primary == "Human Interaction":
            return (
                "Evaluate retention, management support, "
                "career progression, workload, and compensation competitiveness."
            )

        if primary == "Analytical":
            return (
                "Evaluate AI augmentation opportunities, workflow redesign, "
                "knowledge retention, and specialist development."
            )

        if primary == "Digital":
            return (
                "Evaluate workflow automation opportunities, process redesign, "
                "and digital productivity tools."
            )

        if primary == "Physical / Manual":
            return (
                "Evaluate operational process improvements, workforce planning, "
                "safety initiatives, and productivity improvements."
            )

        return (
            "Collect additional occupation-specific evidence before intervention."
        )

    def build_reason(row):

        return (
            f"Cost Share={row['share_of_total_cost_pct']:.1f}% | "
            f"Primary Work={row['primary_work_type']}"
        )

    df["management_focus"] = (
        df.apply(
            build_action,
            axis=1,
        )
    )

    df["evidence"] = (
        df.apply(
            build_reason,
            axis=1,
        )
    )

    executive_table = df[
        [
            "priority_rank",
            "matched_onet_title",
            "share_of_total_cost_pct",
            "total_expected_attrition_cost",
            "primary_work_type",
            "management_focus",
            "evidence",
        ]
    ]

    top_row = executive_table.iloc[0]

    report = ExecutiveReport(
        total_expected_attrition_cost=total_cost,
        top_cost_driver=str(
            top_row["matched_onet_title"]
        ),
        top_cost_share=float(
            top_row["share_of_total_cost_pct"]
        ),
        focus_areas_identified=len(
            executive_table
        ),
        warnings=warnings,
        errors=errors,
    )

    return executive_table, report
