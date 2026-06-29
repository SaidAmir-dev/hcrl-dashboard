"""
HCRL Executive Intelligence Brief Engine

Purpose:
Generate a concise executive-level narrative from Workforce Action Intelligence.

No causal claims.
No ROI estimates.
No guaranteed savings.
No automated employment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class ExecutiveBriefReport:
    briefs_generated: int
    warnings: List[str]
    errors: List[str]


def _money(value) -> str:
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "Not available"


def _pct(value) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "Not available"


def _clean_text(value) -> str:
    if pd.isna(value):
        return "Not available"
    return str(value)


def build_executive_intelligence_brief(
    action_df: pd.DataFrame,
    investigation_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, ExecutiveBriefReport]:

    warnings: List[str] = []
    errors: List[str] = []

    required_action_cols = [
        "action_rank",
        "driver_group",
        "intervention_area",
        "actionability",
        "evidence_strength",
        "management_attention",
        "evidence_drivers",
        "supporting_variables",
        "exposure_linked_to_intervention_area",
        "executive_summary",
    ]

    missing = [
        col for col in required_action_cols
        if col not in action_df.columns
    ]

    if missing:
        errors.append(f"Missing action columns: {missing}")
        return pd.DataFrame(), ExecutiveBriefReport(0, warnings, errors)

    if action_df.empty:
        errors.append("Action table is empty.")
        return pd.DataFrame(), ExecutiveBriefReport(0, warnings, errors)

    if investigation_df is None or investigation_df.empty:
        warnings.append(
            "Investigation drilldown is unavailable. Briefs will not include segment concentration."
        )
        investigation_df = pd.DataFrame()

    rows = []

    for _, action in action_df.iterrows():

        driver_group = action["driver_group"]

        priority_investigation = pd.DataFrame()

        if not investigation_df.empty and "driver_group" in investigation_df.columns:
            priority_investigation = investigation_df[
                investigation_df["driver_group"] == driver_group
            ].copy()

        concentration_points = []
        top_department = "Not available"
        top_role = "Not available"
        top_level = "Not available"

        if not priority_investigation.empty:

            for dimension in ["Department", "Job Role", "Job Level", "Location", "Manager", "Business Unit", "Team"]:
                dim_table = priority_investigation[
                    priority_investigation["dimension"] == dimension
                ].copy()

                if dim_table.empty:
                    continue

                dim_table = dim_table.sort_values(
                    "allocated_exposure_linked_to_priority",
                    ascending=False,
                )

                top = dim_table.iloc[0]

                point = (
                    f"{dimension}: {_clean_text(top['segment'])} "
                    f"({_money(top['allocated_exposure_linked_to_priority'])} allocated exposure)"
                )

                concentration_points.append(point)

                if dimension == "Department":
                    top_department = _clean_text(top["segment"])
                elif dimension == "Job Role":
                    top_role = _clean_text(top["segment"])
                elif dimension == "Job Level":
                    top_level = _clean_text(top["segment"])

        if concentration_points:
            concentration_text = " | ".join(concentration_points[:5])
        else:
            concentration_text = "No segment concentration available."

        narrative = (
            f"{driver_group} is ranked #{int(action['action_rank'])} because it is supported by "
            f"{int(action['evidence_drivers'])} evidence driver(s) and is linked to "
            f"{_money(action['exposure_linked_to_intervention_area'])} of modeled workforce exposure. "
            f"The strongest visible concentration points are: {concentration_text}. "
            f"Management should begin investigation in these exposed workforce areas before expanding the review company-wide."
        )

        board_summary = (
            f"Priority #{int(action['action_rank'])}: {driver_group}. "
            f"Recommended investigation area: {action['intervention_area']}. "
            f"Evidence strength: {action['evidence_strength']}. "
            f"Management attention: {action['management_attention']}. "
            f"Linked modeled exposure: {_money(action['exposure_linked_to_intervention_area'])}."
        )

        rows.append(
            {
                "brief_rank": int(action["action_rank"]),
                "workforce_priority": driver_group,
                "recommended_investigation_area": action["intervention_area"],
                "actionability": action["actionability"],
                "evidence_strength": action["evidence_strength"],
                "management_attention": action["management_attention"],
                "linked_modeled_exposure": float(action["exposure_linked_to_intervention_area"]),
                "evidence_drivers": int(action["evidence_drivers"]),
                "supporting_variables": action["supporting_variables"],
                "top_department": top_department,
                "top_job_role": top_role,
                "top_job_level": top_level,
                "concentration_points": concentration_text,
                "executive_narrative": narrative,
                "board_summary": board_summary,
                "limitations": (
                    "This brief is evidence-aligned decision support. It does not prove causality, "
                    "estimate ROI, guarantee savings, or prescribe automatic employment decisions."
                ),
            }
        )

    brief_df = pd.DataFrame(rows)

    warnings.append(
        "Executive Intelligence Brief synthesizes modeled exposure, driver evidence, and segment concentration into executive-ready narratives. It is not causal attribution or ROI estimation."
    )

    return brief_df, ExecutiveBriefReport(
        briefs_generated=len(brief_df),
        warnings=warnings,
        errors=errors,
    )
