"""
HCRL Intervention Economics Engine

Purpose:
Link evidence-aligned intervention areas to modeled attrition cost exposure.

No causal claims.
No ROI claims.
No guaranteed savings.
No automatic personnel decisions.
No firing recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class InterventionEconomicsReport:
    intervention_areas_analyzed: int
    total_modeled_exposure: float
    warnings: List[str]
    errors: List[str]


def build_intervention_economics_table(
    intervention_table: pd.DataFrame,
    priority_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, InterventionEconomicsReport]:

    warnings: List[str] = []
    errors: List[str] = []

    intervention_required = [
        "driver_group",
        "intervention_area",
        "evidence_drivers",
        "supporting_variables",
        "intervention_evidence_score",
        "actionability",
        "potential_interventions",
    ]

    priority_required = [
        "total_expected_attrition_cost",
    ]

    missing_intervention = [
        col for col in intervention_required
        if col not in intervention_table.columns
    ]

    missing_priority = [
        col for col in priority_required
        if col not in priority_table.columns
    ]

    if missing_intervention:
        errors.append(
            f"Missing intervention table columns: {missing_intervention}"
        )

    if missing_priority:
        errors.append(
            f"Missing priority table columns: {missing_priority}"
        )

    if errors:
        return pd.DataFrame(), InterventionEconomicsReport(
            intervention_areas_analyzed=0,
            total_modeled_exposure=0.0,
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
        errors.append("Total modeled attrition exposure is zero or unavailable.")
        return pd.DataFrame(), InterventionEconomicsReport(
            intervention_areas_analyzed=0,
            total_modeled_exposure=0.0,
            warnings=warnings,
            errors=errors,
        )

    df = intervention_table.copy()

    df["intervention_evidence_score"] = pd.to_numeric(
        df["intervention_evidence_score"],
        errors="coerce",
    )

    df = df[df["intervention_evidence_score"].notna()].copy()

    if df.empty:
        errors.append("No valid intervention evidence scores available.")
        return pd.DataFrame(), InterventionEconomicsReport(
            intervention_areas_analyzed=0,
            total_modeled_exposure=total_exposure,
            warnings=warnings,
            errors=errors,
        )

    total_score = df["intervention_evidence_score"].sum()

    if total_score <= 0:
        errors.append("Total intervention evidence score is zero or unavailable.")
        return pd.DataFrame(), InterventionEconomicsReport(
            intervention_areas_analyzed=0,
            total_modeled_exposure=total_exposure,
            warnings=warnings,
            errors=errors,
        )

    df["share_of_intervention_evidence"] = (
        df["intervention_evidence_score"] / total_score
    )

    df["exposure_linked_to_intervention_area"] = (
        df["share_of_intervention_evidence"] * total_exposure
    )

    df["economic_attention_score"] = (
        df["exposure_linked_to_intervention_area"]
        * df["intervention_evidence_score"]
    )

    df["economic_summary"] = (
        df["intervention_area"].astype(str)
        + " is linked to approximately $"
        + df["exposure_linked_to_intervention_area"]
        .round(0)
        .astype(int)
        .map(lambda x: f"{x:,}")
        + " of modeled workforce exposure, based on driver evidence allocation."
    )

    df = df.sort_values(
        [
            "actionability",
            "economic_attention_score",
        ],
        ascending=[True, False],
    ).reset_index(drop=True)

    df["economic_rank"] = range(
        1,
        len(df) + 1,
    )

    output = df[
        [
            "economic_rank",
            "driver_group",
            "intervention_area",
            "actionability",
            "evidence_drivers",
            "supporting_variables",
            "intervention_evidence_score",
            "share_of_intervention_evidence",
            "exposure_linked_to_intervention_area",
            "economic_attention_score",
            "potential_interventions",
            "economic_summary",
        ]
    ]

    warnings.append(
        "Intervention economics links modeled attrition exposure to intervention review "
        "areas using driver evidence allocation. It does not estimate causal savings, "
        "ROI, or guaranteed financial impact."
    )

    return output, InterventionEconomicsReport(
        intervention_areas_analyzed=len(output),
        total_modeled_exposure=total_exposure,
        warnings=warnings,
        errors=errors,
    )
