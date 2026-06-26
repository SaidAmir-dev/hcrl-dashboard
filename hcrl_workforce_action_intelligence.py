"""
HCRL Workforce Action Intelligence Engine

Purpose:
Convert workforce driver evidence and modeled exposure into executive-level
management investigation priorities.

No causal claims.
No ROI estimates.
No guaranteed savings.
No automated employment decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class WorkforceActionReport:
    action_areas_identified: int
    warnings: List[str]
    errors: List[str]


ACTIONABLE_SORT = {
    "Actionable": 1,
    "Descriptive": 0,
}


INVESTIGATION_QUESTIONS: Dict[str, List[str]] = {
    "Career Progression": [
        "Are employees remaining too long without promotion?",
        "Are internal mobility paths visible and accessible?",
        "Are promotion patterns consistent across departments?",
        "Are high-performing employees progressing internally?",
    ],
    "Compensation": [
        "Is compensation competitive for exposed roles?",
        "Are there signs of pay compression across job levels?",
        "Are salary increases aligned with retention-sensitive roles?",
        "Are long-term incentives being used effectively?",
    ],
    "Work Environment": [
        "Are low satisfaction scores concentrated in specific teams?",
        "Are employee experience issues concentrated by manager, department, or role?",
        "Do exposed groups report weaker job involvement?",
        "Are workload, recognition, and team climate being reviewed together?",
    ],
    "Manager Stability": [
        "Do exposed workforce groups share unstable manager relationships?",
        "Are manager changes concentrated in high-exposure workforce areas?",
        "Do teams with shorter manager tenure show higher modeled risk?",
        "Should manager continuity be reviewed for critical roles?",
    ],
    "Travel / Commute Burden": [
        "Are commute or travel expectations concentrated in exposed groups?",
        "Can flexibility reduce avoidable workforce friction?",
        "Are location requirements aligned with role needs?",
        "Do travel-heavy roles show higher modeled attrition risk?",
    ],
    "Workload": [
        "Is overtime concentrated in exposed roles or departments?",
        "Are staffing levels aligned with demand?",
        "Are workload spikes creating retention pressure?",
        "Should scheduling or capacity planning be reviewed?",
    ],
    "Training and Development": [
        "Are development opportunities reaching exposed workforce groups?",
        "Are training investments aligned with retention-sensitive roles?",
        "Do employees in exposed areas receive enough skill development?",
        "Are learning pathways connected to career progression?",
    ],
    "Department": [
        "Is workforce exposure concentrated in one department?",
        "Are department-level practices contributing to risk variation?",
        "Do departments differ in manager stability, workload, or compensation patterns?",
        "Should department leaders review local workforce conditions?",
    ],
    "Performance": [
        "Are performance ratings aligned with growth opportunities?",
        "Do performance processes support retention-sensitive employees?",
        "Are high-performing employees receiving progression opportunities?",
        "Should performance and career development be reviewed together?",
    ],
    "Occupation": [
        "Are specific roles carrying disproportionate workforce exposure?",
        "Are role-level risks linked to labor-market pressure?",
        "Are hiring pipelines strong enough for exposed occupations?",
        "Should role-specific workforce planning be reviewed?",
    ],
    "Employee Experience": [
        "Are experienced employees showing retention pressure?",
        "Are tenure patterns linked to career progression issues?",
        "Are long-tenured employees receiving enough development opportunities?",
        "Should employee lifecycle patterns be reviewed?",
    ],
    "Education": [
        "Do educational profiles differ across exposed groups?",
        "Is education acting as a descriptive workforce signal?",
        "Should education be reviewed only as context, not as a direct action lever?",
        "Are skills-development pathways more useful than education categories?",
    ],
}


MANAGEMENT_INVESTIGATIONS: Dict[str, List[str]] = {
    "Career Progression": [
        "Primary: Review promotion timelines across departments.",
        "Secondary: Review internal mobility opportunities before external hiring.",
        "Supporting: Review career ladders and succession planning.",
    ],
    "Compensation": [
        "Primary: Review compensation competitiveness for exposed roles.",
        "Secondary: Review pay progression and salary increase patterns.",
        "Supporting: Review long-term incentives and pay compression.",
    ],
    "Work Environment": [
        "Primary: Review employee experience signals in exposed groups.",
        "Secondary: Review team climate, recognition, and engagement patterns.",
        "Supporting: Review whether issues are concentrated by manager, role, or department.",
    ],
    "Manager Stability": [
        "Primary: Review manager continuity in exposed workforce areas.",
        "Secondary: Review leadership support and span of control.",
        "Supporting: Review manager coaching and team stability.",
    ],
    "Travel / Commute Burden": [
        "Primary: Review flexibility options for exposed groups.",
        "Secondary: Review travel burden and commute requirements.",
        "Supporting: Review location strategy and hybrid-work feasibility.",
    ],
    "Workload": [
        "Primary: Review overtime concentration.",
        "Secondary: Review staffing, scheduling, and workload balance.",
        "Supporting: Review capacity planning for exposed teams.",
    ],
    "Training and Development": [
        "Primary: Review training access for exposed workforce groups.",
        "Secondary: Review skills-development pathways.",
        "Supporting: Connect training programs to internal mobility.",
    ],
    "Department": [
        "Primary: Review department-level workforce conditions.",
        "Secondary: Compare local practices across departments.",
        "Supporting: Review department-specific manager, workload, and compensation patterns.",
    ],
    "Performance": [
        "Primary: Review performance management consistency.",
        "Secondary: Connect performance outcomes to growth opportunities.",
        "Supporting: Review whether high performers receive advancement opportunities.",
    ],
    "Occupation": [
        "Primary: Review role-specific workforce planning.",
        "Secondary: Review hiring pipeline strength for exposed occupations.",
        "Supporting: Review external labor-market pressure and role design.",
    ],
    "Employee Experience": [
        "Primary: Review employee lifecycle patterns.",
        "Secondary: Review retention pressure among experienced employees.",
        "Supporting: Review development opportunities for long-tenured employees.",
    ],
    "Education": [
        "Primary: Treat education as contextual evidence.",
        "Secondary: Review skills-development needs instead of education categories alone.",
        "Supporting: Avoid using education profile as a direct personnel action trigger.",
    ],
}


BUSINESS_RISKS: Dict[str, List[str]] = {
    "Career Progression": [
        "Loss of experienced employees",
        "Weak internal mobility",
        "Higher external hiring dependence",
        "Leadership pipeline weakness",
    ],
    "Compensation": [
        "Retention pressure in competitive roles",
        "Pay compression concerns",
        "Higher replacement cost exposure",
        "Lower perceived reward fairness",
    ],
    "Work Environment": [
        "Lower engagement",
        "Reduced team stability",
        "Higher voluntary turnover pressure",
        "Employee experience deterioration",
    ],
    "Manager Stability": [
        "Leadership continuity risk",
        "Team disruption",
        "Lower trust in management",
        "Higher retention pressure in affected teams",
    ],
    "Travel / Commute Burden": [
        "Avoidable friction for employees",
        "Reduced flexibility competitiveness",
        "Retention pressure in location-sensitive roles",
        "Higher dissatisfaction in travel-heavy work",
    ],
    "Workload": [
        "Burnout risk",
        "Capacity strain",
        "Overtime concentration",
        "Reduced workforce sustainability",
    ],
    "Training and Development": [
        "Skill stagnation",
        "Lower internal mobility",
        "Reduced readiness for future work",
        "Higher disengagement among growth-oriented employees",
    ],
    "Department": [
        "Localized workforce instability",
        "Inconsistent management practices",
        "Uneven retention outcomes",
        "Department-specific operational risk",
    ],
    "Performance": [
        "Misalignment between performance and growth",
        "Reduced motivation",
        "Retention pressure among strong performers",
        "Lower trust in evaluation systems",
    ],
    "Occupation": [
        "Role-level labor supply risk",
        "Hiring pipeline weakness",
        "Critical role instability",
        "Workforce planning gaps",
    ],
    "Employee Experience": [
        "Loss of institutional knowledge",
        "Declining employee loyalty",
        "Lifecycle retention gaps",
        "Higher replacement pressure",
    ],
    "Education": [
        "Skill mismatch risk",
        "Training need visibility",
        "Workforce capability gaps",
        "Misinterpretation if used without context",
    ],
}


def _evidence_strength_label(n_drivers: int) -> str:
    if n_drivers >= 4:
        return "Very High"
    if n_drivers >= 2:
        return "High"
    return "Moderate"


def _management_attention_label(rank: int, actionability: str) -> str:
    if actionability != "Actionable":
        return "Contextual Review"
    if rank <= 2:
        return "Immediate Management Review"
    if rank <= 5:
        return "Near-Term Management Review"
    return "Planned Review"


def _safe_join(values: List[str]) -> str:
    return " | ".join(values)


def build_workforce_action_intelligence(
    intervention_economics_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, WorkforceActionReport]:

    warnings: List[str] = []
    errors: List[str] = []

    required_cols = [
        "driver_group",
        "intervention_area",
        "actionability",
        "evidence_drivers",
        "supporting_variables",
        "intervention_evidence_score",
        "exposure_linked_to_intervention_area",
        "economic_attention_score",
        "potential_interventions",
    ]

    missing = [
        col for col in required_cols
        if col not in intervention_economics_table.columns
    ]

    if missing:
        errors.append(f"Missing required columns: {missing}")
        return pd.DataFrame(), WorkforceActionReport(0, warnings, errors)

    df = intervention_economics_table.copy()

    for col in [
        "evidence_drivers",
        "intervention_evidence_score",
        "exposure_linked_to_intervention_area",
        "economic_attention_score",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        df["driver_group"].notna()
        & df["intervention_area"].notna()
        & df["economic_attention_score"].notna()
    ].copy()

    if df.empty:
        errors.append("No valid intervention economics evidence available.")
        return pd.DataFrame(), WorkforceActionReport(0, warnings, errors)

    df["actionability_sort"] = (
        df["actionability"]
        .map(ACTIONABLE_SORT)
        .fillna(0)
    )

    df = (
        df
        .sort_values(
            [
                "actionability_sort",
                "economic_attention_score",
                "exposure_linked_to_intervention_area",
            ],
            ascending=[False, False, False],
        )
        .drop(columns=["actionability_sort"])
        .reset_index(drop=True)
    )

    df["action_rank"] = range(1, len(df) + 1)

    df["evidence_strength"] = df["evidence_drivers"].apply(
        lambda x: _evidence_strength_label(int(x))
    )

    df["management_attention"] = df.apply(
        lambda row: _management_attention_label(
            int(row["action_rank"]),
            str(row["actionability"]),
        ),
        axis=1,
    )

    df["management_questions"] = df["driver_group"].apply(
        lambda x: _safe_join(
            INVESTIGATION_QUESTIONS.get(
                x,
                ["Review this workforce domain with HR leadership."]
            )
        )
    )

    df["recommended_management_investigations"] = df["driver_group"].apply(
        lambda x: _safe_join(
            MANAGEMENT_INVESTIGATIONS.get(
                x,
                ["Review this workforce domain before making decisions."]
            )
        )
    )

    df["business_risks_if_ignored"] = df["driver_group"].apply(
        lambda x: _safe_join(
            BUSINESS_RISKS.get(
                x,
                ["Potential workforce risk may remain unexplained."]
            )
        )
    )

    df["executive_summary"] = df.apply(
        lambda row: (
            f"{row['driver_group']} appears as a management priority because it is "
            f"supported by {int(row['evidence_drivers'])} evidence driver(s) and is linked "
            f"to approximately ${row['exposure_linked_to_intervention_area']:,.0f} "
            f"of modeled workforce exposure."
        ),
        axis=1,
    )

    output_cols = [
        "action_rank",
        "driver_group",
        "intervention_area",
        "actionability",
        "evidence_strength",
        "management_attention",
        "evidence_drivers",
        "supporting_variables",
        "intervention_evidence_score",
        "exposure_linked_to_intervention_area",
        "economic_attention_score",
        "management_questions",
        "recommended_management_investigations",
        "business_risks_if_ignored",
        "executive_summary",
        "potential_interventions",
    ]

    df = df[output_cols]

    warnings.append(
        "Workforce Action Intelligence converts driver evidence and modeled exposure "
        "into management investigation priorities. It does not estimate ROI, causal savings, "
        "or prescribe mandatory actions."
    )

    return df, WorkforceActionReport(
        action_areas_identified=len(df),
        warnings=warnings,
        errors=errors,
    )
