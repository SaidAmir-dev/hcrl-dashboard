"""
HCRL Workforce Narrative Engine

Purpose:
Generate executive-readable workforce narratives from existing HCRL outputs.

No automated employment decisions.
No firing recommendations.
No arbitrary thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class NarrativeReport:
    narratives_generated: int
    warnings: List[str]
    errors: List[str]


def build_workforce_narratives(
    executive_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, NarrativeReport]:

    warnings: List[str] = []
    errors: List[str] = []

    required_cols = [
        "matched_onet_title",
        "share_of_total_cost_pct",
        "total_expected_attrition_cost",
        "primary_work_type",
        "management_focus",
    ]

    missing = [c for c in required_cols if c not in executive_table.columns]

    if missing:
        errors.append(f"Missing required columns: {missing}")
        return pd.DataFrame(), NarrativeReport(0, warnings, errors)

    df = executive_table.copy()

    def build_narrative(row):

        occupation = row["matched_onet_title"]
        cost_share = row["share_of_total_cost_pct"]
        total_cost = row["total_expected_attrition_cost"]
        primary = row["primary_work_type"]
        focus = row["management_focus"]

        return (
            f"{occupation} accounts for {cost_share:.1f}% of total expected "
            f"attrition cost, representing approximately ${total_cost:,.0f} "
            f"in modeled annual workforce exposure. "
            f"The occupation is characterized primarily by {primary} work. "
            f"Management focus: {focus}"
        )

    df["executive_narrative"] = df.apply(build_narrative, axis=1)

    narrative_table = df[
        [
            "priority_rank",
            "matched_onet_title",
            "share_of_total_cost_pct",
            "total_expected_attrition_cost",
            "primary_work_type",
            "executive_narrative",
        ]
    ]

    return narrative_table, NarrativeReport(
        narratives_generated=len(narrative_table),
        warnings=warnings,
        errors=errors,
    )
