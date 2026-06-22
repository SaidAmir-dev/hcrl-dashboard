"""
HCRL Workforce Opportunity Intelligence Engine

Purpose:
Combine workforce economic exposure with leverage evidence to identify
where management attention may create the largest business opportunity.

No causal claims.
No ROI claims.
No automatic personnel decisions.
No firing recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class WorkforceOpportunityReport:
    opportunities_identified: int
    warnings: List[str]
    errors: List[str]


def build_workforce_opportunity_table(
    priority_table: pd.DataFrame,
    leverage_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, WorkforceOpportunityReport]:

    warnings: List[str] = []
    errors: List[str] = []

    priority_required = [
        "total_expected_attrition_cost",
    ]

    leverage_required = [
        "driver_group",
        "leverage_score",
        "evidence_drivers",
        "supporting_variables",
        "actionability",
    ]

    missing_priority = [
        col for col in priority_required
        if col not in priority_table.columns
    ]

    missing_leverage = [
        col for col in leverage_required
        if col not in leverage_table.columns
    ]

    if missing_priority:
        errors.append(
            f"Missing priority table columns: {missing_priority}"
        )

    if missing_leverage:
        errors.append(
            f"Missing leverage table columns: {missing_leverage}"
        )

    if errors:
        return pd.DataFrame(), WorkforceOpportunityReport(
            opportunities_identified=0,
            warnings=warnings,
            errors=errors,
        )

    total_exposure = float(
        pd.to_numeric(
            priority_table["total_expected_attrition_cost"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    if total_exposure <= 0:
        errors.append(
            "Total expected attrition cost is zero or unavailable."
        )
        return pd.DataFrame(), WorkforceOpportunityReport(
            opportunities_identified=0,
            warnings=warnings,
            errors=errors,
        )

    df = leverage_table.copy()

    df["leverage_score"] = pd.to_numeric(
        df["leverage_score"],
        errors="coerce",
    )

    df = df[df["leverage_score"].notna()].copy()

    if df.empty:
        errors.append(
            "No valid leverage scores available."
        )
        return pd.DataFrame(), WorkforceOpportunityReport(
            opportunities_identified=0,
            warnings=warnings,
            errors=errors,
        )

    total_leverage = df["leverage_score"].sum()

    if total_leverage <= 0:
        errors.append(
            "Total leverage score is zero or unavailable."
        )
        return pd.DataFrame(), WorkforceOpportunityReport(
            opportunities_identified=0,
            warnings=warnings,
            errors=errors,
        )

    df["share_of_leverage"] = (
        df["leverage_score"] / total_leverage
    )

    df["estimated_exposure_linked_to_domain"] = (
        df["share_of_leverage"] * total_exposure
    )

    df["opportunity_score"] = (
        df["estimated_exposure_linked_to_domain"]
        * df["leverage_score"]
    )

    df = df.sort_values(
        "opportunity_score",
        ascending=False,
    ).reset_index(drop=True)

    df["opportunity_rank"] = range(
        1,
        len(df) + 1,
    )

    df["opportunity_summary"] = (
        df["driver_group"].astype(str)
        + " is linked to approximately $"
        + df["estimated_exposure_linked_to_domain"]
        .round(0)
        .astype(int)
        .map(lambda x: f"{x:,}")
        + " of modeled workforce exposure, based on its share of leverage evidence."
    )

    output = df[
        [
            "opportunity_rank",
            "driver_group",
            "actionability",
            "evidence_drivers",
            "supporting_variables",
            "leverage_score",
            "share_of_leverage",
            "estimated_exposure_linked_to_domain",
            "opportunity_score",
            "opportunity_summary",
        ]
    ]

    warnings.append(
        "Opportunity estimates allocate modeled attrition exposure across workforce domains "
        "using leverage evidence. This is not causal attribution, ROI estimation, or a "
        "prescriptive recommendation."
    )

    return output, WorkforceOpportunityReport(
        opportunities_identified=len(output),
        warnings=warnings,
        errors=errors,
    )
