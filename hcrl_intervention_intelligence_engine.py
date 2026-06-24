"""
HCRL Intervention Intelligence Engine

Purpose:
Translate attrition driver evidence into evidence-aligned intervention areas.

No causal claims.
No ROI claims.
No automatic personnel decisions.
No firing recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import math
import pandas as pd


@dataclass
class InterventionIntelligenceReport:
    intervention_areas_identified: int
    warnings: List[str]
    errors: List[str]


INTERVENTION_LIBRARY = {
    "Compensation": {
        "intervention_area": "Compensation Review",
        "potential_interventions": (
            "Compensation benchmarking | Internal pay equity review | "
            "Salary progression review | Long-term incentive review"
        ),
    },
    "Career Progression": {
        "intervention_area": "Career Progression Review",
        "potential_interventions": (
            "Promotion pathway review | Internal mobility review | "
            "Career ladder redesign | Succession planning"
        ),
    },
    "Manager Stability": {
        "intervention_area": "Management Stability Review",
        "potential_interventions": (
            "Manager continuity review | Leadership support | "
            "Span-of-control review | Manager coaching"
        ),
    },
    "Work Environment": {
        "intervention_area": "Work Environment Review",
        "potential_interventions": (
            "Engagement review | Job satisfaction review | "
            "Team climate review | Employee experience diagnostics"
        ),
    },
    "Workload": {
        "intervention_area": "Workload Review",
        "potential_interventions": (
            "Overtime review | Staffing balance review | "
            "Scheduling review | Workflow redesign"
        ),
    },
    "Travel / Commute Burden": {
        "intervention_area": "Travel and Flexibility Review",
        "potential_interventions": (
            "Travel burden review | Commute flexibility review | "
            "Hybrid work review | Location strategy review"
        ),
    },
    "Training and Development": {
        "intervention_area": "Training and Development Review",
        "potential_interventions": (
            "Training access review | Skill development review | "
            "Onboarding review | Learning investment review"
        ),
    },
    "Department": {
        "intervention_area": "Department-Level Review",
        "potential_interventions": (
            "Department risk review | Operating model review | "
            "Management environment review | Function-specific diagnostics"
        ),
    },
    "Occupation": {
        "intervention_area": "Occupation-Level Review",
        "potential_interventions": (
            "Role-specific retention review | Labor-market exposure review | "
            "Hiring pipeline review | Occupation-level workforce planning"
        ),
    },
    "Education": {
        "intervention_area": "Education Profile Review",
        "potential_interventions": (
            "Education mix review | Skill-fit review | "
            "Credential requirements review | Role specialization review"
        ),
    },
    "Performance": {
        "intervention_area": "Performance System Review",
        "potential_interventions": (
            "Performance rating review | Feedback process review | "
            "Reward alignment review | Progression fairness review"
        ),
    },
    "Employee Experience": {
        "intervention_area": "Employee Experience Review",
        "potential_interventions": (
            "Onboarding review | Early-career support | "
            "Experience segmentation | Retention journey review"
        ),
    },
}


def build_intervention_intelligence_table(
    driver_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, InterventionIntelligenceReport]:

    warnings: List[str] = []
    errors: List[str] = []

    required_cols = [
        "driver_group",
        "driver_variable",
        "association_value",
        "actionability",
    ]

    missing = [col for col in required_cols if col not in driver_table.columns]

    if missing:
        errors.append(f"Missing required driver columns: {missing}")
        return pd.DataFrame(), InterventionIntelligenceReport(0, warnings, errors)

    df = driver_table.copy()

    df["association_value"] = pd.to_numeric(
        df["association_value"],
        errors="coerce",
    )

    df = df[df["association_value"].notna()].copy()

    if df.empty:
        errors.append("No valid driver evidence available.")
        return pd.DataFrame(), InterventionIntelligenceReport(0, warnings, errors)

    df["absolute_association"] = df["association_value"].abs()

    grouped = (
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

    grouped["intervention_evidence_score"] = (
        grouped["average_association"]
        * grouped["evidence_drivers"].apply(math.sqrt)
    )

    grouped["actionability_priority"] = grouped["actionability"].map(
        {
            "Actionable": 1,
            "Descriptive": 0,
        }
    ).fillna(0)

    grouped = grouped.sort_values(
        [
            "actionability_priority",
            "intervention_evidence_score",
            "strongest_association",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    rows = []

    for _, row in grouped.iterrows():

        driver_group = str(row["driver_group"])

        library_entry = INTERVENTION_LIBRARY.get(
            driver_group,
            {
                "intervention_area": "Further Investigation",
                "potential_interventions": (
                    "Additional analysis required before identifying intervention categories."
                ),
            },
        )

        rows.append(
            {
                "driver_group": driver_group,
                "intervention_area": library_entry["intervention_area"],
                "evidence_drivers": int(row["evidence_drivers"]),
                "supporting_variables": row["supporting_variables"],
                "strongest_association": float(row["strongest_association"]),
                "average_association": float(row["average_association"]),
                "intervention_evidence_score": float(
                    row["intervention_evidence_score"]
                ),
                "actionability": row["actionability"],
                "potential_interventions": library_entry["potential_interventions"],
                "decision_note": (
                    "These are evidence-aligned intervention categories for management "
                    "review. They are not causal prescriptions or automatic decisions."
                ),
            }
        )

    output = pd.DataFrame(rows)

    output["intervention_rank"] = range(
        1,
        len(output) + 1,
    )

    output = output[
        [
            "intervention_rank",
            "driver_group",
            "intervention_area",
            "evidence_drivers",
            "supporting_variables",
            "strongest_association",
            "average_association",
            "intervention_evidence_score",
            "actionability",
            "potential_interventions",
            "decision_note",
        ]
    ]

    warnings.append(
        "Intervention intelligence translates statistical driver evidence into "
        "management review areas. It does not estimate causality, ROI, or guaranteed savings."
    )

    return output, InterventionIntelligenceReport(
        intervention_areas_identified=len(output),
        warnings=warnings,
        errors=errors,
    )
