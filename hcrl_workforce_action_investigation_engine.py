"""
HCRL Workforce Action Investigation Engine

Purpose:
Convert intervention economics into executive action priorities and identify
where inside the company management should begin investigation.

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
class WorkforceActionInvestigationReport:
    action_areas_identified: int
    investigation_rows_generated: int
    warnings: List[str]
    errors: List[str]


ACTIONABLE_SORT = {
    "Actionable": 1,
    "Descriptive": 0,
}


MANAGEMENT_QUESTIONS = {
    "Career Progression": [
        "Are employees remaining too long without promotion?",
        "Are promotion patterns consistent across departments?",
        "Are internal mobility paths visible and accessible?",
        "Are high-performing employees progressing internally?",
    ],
    "Compensation": [
        "Is compensation competitive for exposed roles?",
        "Are there signs of pay compression across job levels?",
        "Are salary increases aligned with retention-sensitive roles?",
        "Are long-term incentives being used effectively?",
    ],
    "Work Environment": [
        "Are employee experience issues concentrated by manager, department, or role?",
        "Do exposed groups report weaker job involvement?",
        "Are satisfaction-related signals concentrated in specific teams?",
        "Are workload, recognition, and team climate being reviewed together?",
    ],
    "Manager Stability": [
        "Are manager relationship issues concentrated in exposed workforce areas?",
        "Do teams with shorter manager tenure show higher modeled risk?",
        "Are manager changes concentrated in exposed groups?",
        "Should manager continuity be reviewed for critical roles?",
    ],
    "Workload": [
        "Is overtime concentrated in exposed roles or departments?",
        "Are staffing levels aligned with demand?",
        "Are workload spikes creating retention pressure?",
        "Should scheduling or capacity planning be reviewed?",
    ],
    "Travel / Commute Burden": [
        "Are travel or commute expectations concentrated in exposed groups?",
        "Can flexibility reduce avoidable workforce friction?",
        "Are location requirements aligned with role needs?",
        "Do travel-heavy roles show higher modeled attrition risk?",
    ],
    "Training and Development": [
        "Are development opportunities reaching exposed workforce groups?",
        "Are training investments aligned with retention-sensitive roles?",
        "Are learning pathways connected to career progression?",
        "Do exposed employees receive enough skill development?",
    ],
    "Department": [
        "Is exposure concentrated in specific departments?",
        "Are department-level practices contributing to risk variation?",
        "Do departments differ in manager stability, workload, or compensation patterns?",
        "Should department leaders review local workforce conditions?",
    ],
    "Occupation": [
        "Are specific roles carrying disproportionate workforce exposure?",
        "Are role-level risks linked to labor market pressure?",
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
        "Is education acting as a contextual workforce signal?",
        "Are skills-development pathways more useful than education categories?",
        "Do educational profiles differ across exposed groups?",
        "Should education be reviewed only as context, not as a direct action lever?",
    ],
    "Performance": [
        "Are performance ratings aligned with growth opportunities?",
        "Do performance processes support retention-sensitive employees?",
        "Are high-performing employees receiving progression opportunities?",
        "Should performance and career development be reviewed together?",
    ],
}


MANAGEMENT_INVESTIGATIONS = {
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
    "Workload": [
        "Primary: Review overtime concentration.",
        "Secondary: Review staffing, scheduling, and workload balance.",
        "Supporting: Review capacity planning for exposed teams.",
    ],
    "Travel / Commute Burden": [
        "Primary: Review flexibility options for exposed groups.",
        "Secondary: Review travel burden and commute requirements.",
        "Supporting: Review location strategy and hybrid-work feasibility.",
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
    "Performance": [
        "Primary: Review performance management consistency.",
        "Secondary: Connect performance outcomes to growth opportunities.",
        "Supporting: Review whether high performers receive advancement opportunities.",
    ],
}


BUSINESS_RISKS = {
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
    "Workload": [
        "Burnout risk",
        "Capacity strain",
        "Overtime concentration",
        "Reduced workforce sustainability",
    ],
    "Travel / Commute Burden": [
        "Avoidable employee friction",
        "Reduced flexibility competitiveness",
        "Retention pressure in location-sensitive roles",
        "Higher dissatisfaction in travel-heavy work",
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
    "Performance": [
        "Misalignment between performance and growth",
        "Reduced motivation",
        "Retention pressure among strong performers",
        "Lower trust in evaluation systems",
    ],
}


DIMENSION_CANDIDATES = {
    "Department": ["department", "Department"],
    "Job Role": ["job_title", "JobRole", "matched_onet_title"],
    "Location": ["location", "Location"],
    "Job Level": ["JobLevel", "job_level", "level"],
    "Manager": ["manager", "Manager", "manager_id", "ManagerID", "manager_name"],
}


def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


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


def _join(values: List[str]) -> str:
    return " | ".join(values)


def _build_action_table(
    intervention_economics_table: pd.DataFrame,
) -> pd.DataFrame:

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

    df["actionability_sort"] = (
        df["actionability"]
        .map(ACTIONABLE_SORT)
        .fillna(0)
    )

    df = (
        df.sort_values(
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
        lambda x: _join(
            MANAGEMENT_QUESTIONS.get(
                x,
                ["Review this workforce domain with HR leadership."]
            )
        )
    )

    df["recommended_management_investigations"] = df["driver_group"].apply(
        lambda x: _join(
            MANAGEMENT_INVESTIGATIONS.get(
                x,
                ["Review this workforce domain before making decisions."]
            )
        )
    )

    df["business_risks_if_ignored"] = df["driver_group"].apply(
        lambda x: _join(
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

    return df


def _build_investigation_table(
    workforce_df: pd.DataFrame,
    action_df: pd.DataFrame,
    top_n_per_dimension: int = 5,
) -> pd.DataFrame:

    required = [
        "expected_attrition_cost",
        "predicted_attrition_probability",
    ]

    if any(col not in workforce_df.columns for col in required):
        return pd.DataFrame()

    df = workforce_df.copy()

    df["expected_attrition_cost"] = pd.to_numeric(
        df["expected_attrition_cost"],
        errors="coerce",
    ).fillna(0)

    df["predicted_attrition_probability"] = pd.to_numeric(
        df["predicted_attrition_probability"],
        errors="coerce",
    )

    total_company_exposure = float(df["expected_attrition_cost"].sum())

    if total_company_exposure <= 0:
        return pd.DataFrame()

    rows = []

    for _, action_row in action_df.iterrows():

        driver_group = action_row["driver_group"]
        linked_exposure = float(
            action_row["exposure_linked_to_intervention_area"]
        )

        for dimension_name, candidates in DIMENSION_CANDIDATES.items():

            dim_col = _first_existing_column(df, candidates)

            if dim_col is None:
                continue

            temp = df[
                df[dim_col].notna()
            ].copy()

            if temp.empty:
                continue

            grouped = (
                temp.groupby(dim_col)
                .agg(
                    employees=(dim_col, "count"),
                    avg_predicted_attrition_probability=(
                        "predicted_attrition_probability",
                        "mean",
                    ),
                    total_segment_exposure=(
                        "expected_attrition_cost",
                        "sum",
                    ),
                )
                .reset_index()
                .rename(columns={dim_col: "segment"})
            )

            grouped["share_of_company_exposure"] = (
                grouped["total_segment_exposure"] / total_company_exposure
            )

            grouped["allocated_exposure_linked_to_priority"] = (
                grouped["share_of_company_exposure"] * linked_exposure
            )

            grouped = grouped.sort_values(
                "allocated_exposure_linked_to_priority",
                ascending=False,
            ).head(top_n_per_dimension)

            for _, g in grouped.iterrows():
                rows.append(
                    {
                        "driver_group": driver_group,
                        "dimension": dimension_name,
                        "source_column": dim_col,
                        "segment": g["segment"],
                        "employees": int(g["employees"]),
                        "avg_predicted_attrition_probability": float(
                            g["avg_predicted_attrition_probability"]
                        ),
                        "total_segment_exposure": float(
                            g["total_segment_exposure"]
                        ),
                        "share_of_company_exposure": float(
                            g["share_of_company_exposure"]
                        ),
                        "allocated_exposure_linked_to_priority": float(
                            g["allocated_exposure_linked_to_priority"]
                        ),
                    }
                )

    return pd.DataFrame(rows)


def build_workforce_action_investigation(
    workforce_df: pd.DataFrame,
    intervention_economics_table: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, WorkforceActionInvestigationReport]:

    warnings: List[str] = []
    errors: List[str] = []

    required_intervention_cols = [
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
        col for col in required_intervention_cols
        if col not in intervention_economics_table.columns
    ]

    if missing:
        errors.append(f"Missing intervention economics columns: {missing}")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            WorkforceActionInvestigationReport(0, 0, warnings, errors),
        )

    if intervention_economics_table.empty:
        errors.append("Intervention economics table is empty.")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            WorkforceActionInvestigationReport(0, 0, warnings, errors),
        )

    action_df = _build_action_table(intervention_economics_table)

    if action_df.empty:
        errors.append("No valid workforce action priorities could be generated.")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            WorkforceActionInvestigationReport(0, 0, warnings, errors),
        )

    investigation_df = _build_investigation_table(
        workforce_df=workforce_df,
        action_df=action_df,
    )

    if investigation_df.empty:
        warnings.append(
            "Investigation drill-down could not be generated. This usually means "
            "expected_attrition_cost, predicted_attrition_probability, or usable "
            "segment columns are missing."
        )

    warnings.append(
        "Workforce Action Investigation identifies where management may begin review "
        "based on modeled exposure concentration. Allocated exposure is not causal "
        "attribution, ROI, or guaranteed savings."
    )

    return (
        action_df,
        investigation_df,
        WorkforceActionInvestigationReport(
            action_areas_identified=len(action_df),
            investigation_rows_generated=len(investigation_df),
            warnings=warnings,
            errors=errors,
        ),
    )
