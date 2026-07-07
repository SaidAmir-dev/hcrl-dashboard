from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


@dataclass
class ExecutiveBriefReport:
    briefs_generated: int
    warnings: List[str]
    errors: List[str]


DOMAIN_LANGUAGE: Dict[str, Dict[str, object]] = {
    "Career Progression": {
        "finding": "Promotion-related workforce signals represent the largest visible management investigation area.",
        "primary_question": "Are promotion and internal mobility pathways aligned with exposed workforce groups?",
        "workflow": [
            "Validate promotion evidence",
            "Review exposed department",
            "Compare role tenure",
            "Assess internal mobility",
            "Prepare leadership findings",
        ],
        "questions": [
            "Are employees remaining too long without role progression?",
            "Are promotion timelines consistent across departments and job levels?",
            "Are internal mobility paths visible and accessible?",
            "Are high-performing employees progressing internally?",
        ],
        "focus": [
            "Review promotion timelines across exposed groups.",
            "Compare tenure and current-role duration by department and job level.",
            "Assess whether internal mobility is used before external hiring.",
            "Review career ladders and succession planning for exposed roles.",
        ],
    },
    "Compensation": {
        "finding": "Compensation-related signals indicate a management review area for exposed workforce groups.",
        "primary_question": "Is compensation aligned with retention-sensitive roles and exposed workforce groups?",
        "workflow": [
            "Validate compensation evidence",
            "Review exposed roles",
            "Check pay progression",
            "Assess pay compression",
            "Prepare compensation findings",
        ],
        "questions": [
            "Is compensation competitive for exposed roles?",
            "Are salary increases aligned with retention-sensitive groups?",
            "Are there signs of pay compression across job levels?",
            "Are long-term incentives being used effectively?",
        ],
        "focus": [
            "Review compensation competitiveness for exposed roles.",
            "Compare pay progression by role, level, and department.",
            "Assess pay compression and salary increase patterns.",
            "Review long-term incentive structures for exposed groups.",
        ],
    },
    "Work Environment": {
        "finding": "Employee experience and work-environment signals show a visible management review area.",
        "primary_question": "Are employee experience issues concentrated in exposed teams, roles, or departments?",
        "workflow": [
            "Validate experience evidence",
            "Compare exposed teams",
            "Assess engagement signals",
            "Review manager climate",
            "Prepare experience findings",
        ],
        "questions": [
            "Are low satisfaction signals concentrated in specific teams?",
            "Do exposed groups report weaker job involvement?",
            "Are experience issues concentrated by manager, department, or role?",
            "Are workload, recognition, and team climate being reviewed together?",
        ],
        "focus": [
            "Review employee experience signals in exposed groups.",
            "Compare job satisfaction and involvement across departments.",
            "Assess team climate and recognition patterns.",
            "Review overlap with workload or manager signals.",
        ],
    },
    "Manager Stability": {
        "finding": "Manager stability signals indicate a leadership-continuity review area.",
        "primary_question": "Are manager relationships or leadership continuity issues concentrated in exposed groups?",
        "workflow": [
            "Validate manager evidence",
            "Review exposed teams",
            "Compare manager continuity",
            "Assess span of control",
            "Prepare management findings",
        ],
        "questions": [
            "Do exposed groups share unstable manager relationships?",
            "Are manager changes concentrated in exposed workforce areas?",
            "Do teams with shorter manager tenure show higher modeled risk?",
            "Should manager continuity be reviewed for critical roles?",
        ],
        "focus": [
            "Review manager continuity in exposed workforce areas.",
            "Compare teams by manager relationship duration.",
            "Assess leadership support and span of control.",
            "Review manager coaching and team stability.",
        ],
    },
    "Workload": {
        "finding": "Workload signals indicate a capacity and staffing review area.",
        "primary_question": "Are workload demands concentrated in exposed workforce groups?",
        "workflow": [
            "Validate workload evidence",
            "Compare overtime concentration",
            "Assess staffing levels",
            "Review scheduling pressure",
            "Prepare workload findings",
        ],
        "questions": [
            "Is overtime concentrated in exposed roles or departments?",
            "Are workload spikes aligned with elevated modeled risk?",
            "Are staffing levels aligned with demand?",
            "Should scheduling or capacity planning be reviewed?",
        ],
        "focus": [
            "Review overtime and workload patterns in exposed groups.",
            "Compare staffing levels against demand-sensitive roles.",
            "Assess scheduling and capacity planning.",
            "Review overlap with work-environment signals.",
        ],
    },
    "Travel / Commute Burden": {
        "finding": "Travel and commute signals indicate a flexibility and location review area.",
        "primary_question": "Are commute, travel, or location requirements contributing to exposed workforce pressure?",
        "workflow": [
            "Validate travel evidence",
            "Compare exposed locations",
            "Assess flexibility options",
            "Review role-location fit",
            "Prepare flexibility findings",
        ],
        "questions": [
            "Are commute or travel expectations concentrated in exposed groups?",
            "Can flexibility reduce avoidable workforce friction?",
            "Are location requirements aligned with role needs?",
            "Do travel-heavy roles show higher modeled attrition risk?",
        ],
        "focus": [
            "Review flexibility options for exposed groups.",
            "Compare travel burden and commute requirements across roles.",
            "Assess role-location alignment.",
            "Review location strategy and hybrid-work feasibility.",
        ],
    },
    "Training and Development": {
        "finding": "Training and development signals indicate a skills and mobility review area.",
        "primary_question": "Are development opportunities aligned with exposed workforce groups?",
        "workflow": [
            "Validate development evidence",
            "Review training access",
            "Compare exposed roles",
            "Connect learning to mobility",
            "Prepare development findings",
        ],
        "questions": [
            "Are development opportunities reaching exposed workforce groups?",
            "Are training investments aligned with retention-sensitive roles?",
            "Are learning pathways connected to career progression?",
            "Do exposed employees receive enough skill development?",
        ],
        "focus": [
            "Review training access for exposed groups.",
            "Compare training participation across departments and roles.",
            "Assess skills-development pathways.",
            "Connect training programs to internal mobility.",
        ],
    },
}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _money(value) -> str:
    return f"${_safe_float(value):,.0f}"


def _md_money(value) -> str:
    return _money(value).replace("$", r"\$")


def _clean(value) -> str:
    if value is None:
        return "Not available"
    try:
        if pd.isna(value):
            return "Not available"
    except Exception:
        pass

    text = str(value).strip()
    return text if text else "Not available"


def _format_segment(value) -> str:
    text = _clean(value)
    if text.endswith(".0"):
        return text[:-2]
    return text


def _split_pipe(value) -> List[str]:
    text = _clean(value)
    if text == "Not available":
        return []
    return [x.strip() for x in text.split("|") if x.strip()]


def _domain_config(domain: str) -> Dict[str, object]:
    return DOMAIN_LANGUAGE.get(
        domain,
        {
            "finding": f"{domain} appears as a workforce investigation area.",
            "primary_question": f"What explains the workforce exposure associated with {domain}?",
            "workflow": [
                "Validate evidence",
                "Compare exposed segments",
                "Review operating context",
                "Discuss with leaders",
                "Prepare findings",
            ],
            "questions": [
                f"Where is {domain} exposure concentrated?",
                "Which workforce groups contribute most to this signal?",
                "Does the evidence align with management observations?",
                "What additional context is needed before action?",
            ],
            "focus": [
                f"Review {domain} in exposed workforce groups.",
                "Compare concentration by department, role, and level.",
                "Validate findings with HR and business leadership.",
                "Prepare a management investigation summary.",
            ],
        },
    )


def _star_rating(label: str) -> str:
    if label == "Very High":
        return "★★★★★"
    if label == "High":
        return "★★★★☆"
    if label == "Moderate":
        return "★★★☆☆"
    return "★★☆☆☆"


def _attention_badge(label: str) -> str:
    if label == "Immediate Management Review":
        return "🔴 Immediate"
    if label == "Near-Term Management Review":
        return "🟠 Near-Term"
    if label == "Planned Review":
        return "🟡 Planned"
    return "🔵 Contextual"


def _card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
<div style="
    border:1px solid #e5e7eb;
    border-radius:18px;
    padding:20px;
    background:#ffffff;
    box-shadow:0 1px 4px rgba(0,0,0,0.06);
    min-height:130px;
">
    <div style="font-size:13px;color:#667085;margin-bottom:8px;">{title}</div>
    <div style="font-size:25px;font-weight:750;color:#111827;line-height:1.2;">{value}</div>
    <div style="font-size:13px;color:#667085;margin-top:8px;">{subtitle}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_step_flow(steps: List[str]) -> None:
    if not steps:
        return

    cols = st.columns(len(steps))

    for idx, step in enumerate(steps):
        with cols[idx]:
            st.markdown(
                f"""
<div style="
    border:1px solid #d9dee7;
    border-radius:14px;
    padding:16px;
    min-height:108px;
    background:#ffffff;
    box-shadow:0 1px 2px rgba(0,0,0,0.04);
">
    <div style="font-size:13px;color:#667085;margin-bottom:6px;">
        Step {idx + 1}
    </div>
    <div style="font-size:16px;font-weight:650;color:#1f2937;">
        {step}
    </div>
</div>
                """,
                unsafe_allow_html=True,
            )


def _investigation_points(
    investigation_df: pd.DataFrame,
    domain: str,
) -> Tuple[List[str], Dict[str, str], Dict[str, float]]:
    points: List[str] = []
    top_by_dimension: Dict[str, str] = {}
    exposure_by_dimension: Dict[str, float] = {}

    if investigation_df is None or investigation_df.empty:
        return points, top_by_dimension, exposure_by_dimension

    if "driver_group" not in investigation_df.columns:
        return points, top_by_dimension, exposure_by_dimension

    required = ["dimension", "segment", "allocated_exposure_linked_to_priority"]

    if any(col not in investigation_df.columns for col in required):
        return points, top_by_dimension, exposure_by_dimension

    df = investigation_df[
        investigation_df["driver_group"].astype(str) == str(domain)
    ].copy()

    if df.empty:
        return points, top_by_dimension, exposure_by_dimension

    df["allocated_exposure_linked_to_priority"] = pd.to_numeric(
        df["allocated_exposure_linked_to_priority"],
        errors="coerce",
    ).fillna(0)

    dimensions = [
        "Department",
        "Job Role",
        "Job Level",
        "Location",
        "Manager",
        "Business Unit",
        "Team",
        "Cost Center",
    ]

    for dimension in dimensions:
        dim_table = df[df["dimension"].astype(str) == dimension].copy()

        if dim_table.empty:
            continue

        dim_table = dim_table.sort_values(
            "allocated_exposure_linked_to_priority",
            ascending=False,
        )

        top = dim_table.iloc[0]
        segment = _format_segment(top["segment"])
        exposure = _safe_float(top["allocated_exposure_linked_to_priority"])

        top_by_dimension[dimension] = segment
        exposure_by_dimension[dimension] = exposure
        points.append(f"{dimension}: {segment} ({_money(exposure)} allocated exposure)")

    return points, top_by_dimension, exposure_by_dimension


def build_executive_intelligence_brief(
    action_df: pd.DataFrame,
    investigation_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, ExecutiveBriefReport]:

    warnings: List[str] = []
    errors: List[str] = []

    if action_df is None or action_df.empty:
        errors.append("Workforce Action Intelligence output is required.")
        return pd.DataFrame(), ExecutiveBriefReport(0, warnings, errors)

    required = [
        "action_rank",
        "driver_group",
        "intervention_area",
        "actionability",
        "evidence_strength",
        "management_attention",
        "evidence_drivers",
        "supporting_variables",
        "exposure_linked_to_intervention_area",
    ]

    missing = [col for col in required if col not in action_df.columns]

    if missing:
        errors.append(f"Missing required Action Intelligence columns: {missing}")
        return pd.DataFrame(), ExecutiveBriefReport(0, warnings, errors)

    if investigation_df is None:
        investigation_df = pd.DataFrame()

    rows = []

    for _, action in action_df.iterrows():
        domain = _clean(action["driver_group"])
        config = _domain_config(domain)

        concentration_points, top_by_dimension, exposure_by_dimension = _investigation_points(
            investigation_df,
            domain,
        )

        linked_exposure = _safe_float(action["exposure_linked_to_intervention_area"])
        evidence_drivers = _safe_int(action["evidence_drivers"])
        evidence_strength = _clean(action["evidence_strength"])
        management_attention = _clean(action["management_attention"])

        concentration_sentence = (
            "; ".join(concentration_points[:4])
            if concentration_points
            else "segment concentration is not available from the uploaded data"
        )

        executive_assessment = (
            f"{domain} represents Priority #{_safe_int(action['action_rank'])}. "
            f"The uploaded workforce data shows {evidence_drivers} supporting evidence signal(s) "
            f"linked to approximately {_money(linked_exposure)} of modeled workforce exposure. "
            f"The most visible concentration points are {concentration_sentence}. "
            f"Leadership should begin investigation in these exposed workforce segments before expanding the review company-wide."
        )

        executive_story = (
            f"{domain} is the leading workforce investigation area. "
            f"Evidence is strongest in {top_by_dimension.get('Department', 'the highest-exposure organizational areas')}, "
            f"with role-level concentration around {top_by_dimension.get('Job Role', 'the most exposed workforce roles')}. "
            f"The evidence should be reviewed through the lens of: {config['primary_question']}"
        )

        board_summary = (
            f"{domain}: {_money(linked_exposure)} linked exposure, "
            f"{evidence_strength} evidence strength, {management_attention}."
        )

        rows.append(
            {
                "brief_rank": _safe_int(action["action_rank"]),
                "workforce_priority": domain,
                "executive_finding": config["finding"],
                "executive_story": executive_story,
                "recommended_investigation_area": _clean(action["intervention_area"]),
                "primary_management_question": config["primary_question"],
                "actionability": _clean(action["actionability"]),
                "evidence_strength": evidence_strength,
                "evidence_rating": _star_rating(evidence_strength),
                "management_attention": management_attention,
                "attention_badge": _attention_badge(management_attention),
                "linked_modeled_exposure": linked_exposure,
                "evidence_drivers": evidence_drivers,
                "supporting_variables": _clean(action["supporting_variables"]),
                "top_department": top_by_dimension.get("Department", "Not available"),
                "top_department_exposure": exposure_by_dimension.get("Department", 0.0),
                "top_job_role": top_by_dimension.get("Job Role", "Not available"),
                "top_job_role_exposure": exposure_by_dimension.get("Job Role", 0.0),
                "top_job_level": top_by_dimension.get("Job Level", "Not available"),
                "top_job_level_exposure": exposure_by_dimension.get("Job Level", 0.0),
                "top_location": top_by_dimension.get("Location", "Not available"),
                "concentration_points": " | ".join(concentration_points),
                "executive_assessment": executive_assessment,
                "board_summary": board_summary,
                "management_questions": " | ".join(config["questions"]),
                "review_actions": " | ".join(config["focus"]),
                "investigation_workflow": " | ".join(config["workflow"]),
                "limitations": (
                    "This brief is evidence-aligned decision support. It does not prove causality, "
                    "estimate ROI, guarantee savings, or prescribe automatic employment decisions."
                ),
            }
        )

    brief_df = pd.DataFrame(rows).sort_values(
        ["brief_rank", "linked_modeled_exposure"],
        ascending=[True, False],
    ).reset_index(drop=True)

    warnings.append(
        "Executive Intelligence Brief synthesizes modeled exposure, driver evidence, "
        "and segment concentration into executive-ready narratives. It is not causal "
        "attribution, ROI estimation, or an automated employment decision."
    )

    return brief_df, ExecutiveBriefReport(len(brief_df), warnings, errors)


def _render_top_priority_chart(executive_brief_df: pd.DataFrame) -> None:
    chart_df = executive_brief_df[
        ["workforce_priority", "linked_modeled_exposure"]
    ].copy()

    chart_df = chart_df.sort_values(
        "linked_modeled_exposure",
        ascending=False,
    ).head(8)

    chart_df = chart_df.set_index("workforce_priority")

    st.bar_chart(chart_df)


def _render_dimension_chart(
    investigation_df: pd.DataFrame,
    selected_priority: str,
    dimension: str,
) -> None:
    if investigation_df is None or investigation_df.empty:
        return

    df = investigation_df[
        (investigation_df["driver_group"].astype(str) == str(selected_priority))
        & (investigation_df["dimension"].astype(str) == dimension)
    ].copy()

    if df.empty:
        return

    df["allocated_exposure_linked_to_priority"] = pd.to_numeric(
        df["allocated_exposure_linked_to_priority"],
        errors="coerce",
    ).fillna(0)

    df = df.sort_values(
        "allocated_exposure_linked_to_priority",
        ascending=False,
    ).head(8)

    chart_df = df[
        ["segment", "allocated_exposure_linked_to_priority"]
    ].copy()

    chart_df["segment"] = chart_df["segment"].astype(str)
    chart_df = chart_df.set_index("segment")

    st.bar_chart(chart_df)


def _render_investigation_tree(selected: pd.Series) -> None:
    nodes = [
        selected["workforce_priority"],
        selected["top_department"],
        selected["top_job_role"],
        f"Job Level {selected['top_job_level']}",
        _money(selected["linked_modeled_exposure"]),
    ]

    nodes = [x for x in nodes if _clean(x) != "Not available"]

    st.markdown("### Investigation Path")

    html = "<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>"

    for idx, node in enumerate(nodes):
        html += f"""
<div style="
    border:1px solid #d1d5db;
    border-radius:999px;
    padding:10px 16px;
    background:#ffffff;
    font-weight:650;
    color:#111827;
    box-shadow:0 1px 2px rgba(0,0,0,0.05);
">
    {node}
</div>
        """
        if idx < len(nodes) - 1:
            html += "<div style='font-size:22px;color:#667085;'>→</div>"

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)

def _render_executive_explainability(
    selected: pd.Series,
    investigation_df: Optional[pd.DataFrame],
) -> None:

    st.subheader("Why This Is Priority #1")

    priority = selected["workforce_priority"]
    linked_exposure = _money(selected["linked_modeled_exposure"])
    evidence_drivers = int(selected["evidence_drivers"])
    supporting_variables = selected["supporting_variables"]

    st.markdown(
        f"""
<div style="
    background:#fff7ed;
    border-left:6px solid #f97316;
    border-radius:16px;
    padding:22px;
    font-size:17px;
    line-height:1.6;
    margin-bottom:18px;
">
<strong>{priority}</strong> is ranked as a top workforce priority because it combines
<strong>{linked_exposure}</strong> of modeled workforce exposure with
<strong>{evidence_drivers}</strong> supporting workforce evidence signal(s).

<br><br>

<strong>Supporting evidence variables:</strong><br>
{supporting_variables}
</div>
        """,
        unsafe_allow_html=True,
    )

    if investigation_df is None or investigation_df.empty:
        st.info("No segment-level explanation is available for this priority.")
        return

    subset = investigation_df[
        investigation_df["driver_group"].astype(str) == str(priority)
    ].copy()

    if subset.empty:
        st.info("No segment-level explanation is available for this priority.")
        return

    subset["allocated_exposure_linked_to_priority"] = pd.to_numeric(
        subset["allocated_exposure_linked_to_priority"],
        errors="coerce",
    ).fillna(0)

    explanation_rows = []

    for dimension in subset["dimension"].dropna().unique():

        dim_df = subset[
            subset["dimension"].astype(str) == str(dimension)
        ].copy()

        if dim_df.empty:
            continue

        dim_df = dim_df.sort_values(
            "allocated_exposure_linked_to_priority",
            ascending=False,
        )

        top_row = dim_df.iloc[0]
        total_dim_exposure = dim_df["allocated_exposure_linked_to_priority"].sum()

        share = 0.0
        if total_dim_exposure > 0:
            share = (
                top_row["allocated_exposure_linked_to_priority"]
                / total_dim_exposure
            )

        explanation_rows.append(
            {
                "Dimension": dimension,
                "Top Segment": top_row["segment"],
                "Employees": int(top_row["employees"]),
                "Allocated Exposure": _money(
                    top_row["allocated_exposure_linked_to_priority"]
                ),
                "Share Within Dimension": f"{share:.1%}",
            }
        )

    if explanation_rows:

        explanation_df = pd.DataFrame(explanation_rows)

        st.markdown("### Main Evidence Concentration Points")

        st.dataframe(
            explanation_df,
            use_container_width=True,
        )

        top_points = explanation_df.head(3)

        bullets = []
        for _, row in top_points.iterrows():
            bullets.append(
                f"- **{row['Dimension']}**: {row['Top Segment']} "
                f"({row['Allocated Exposure']}, {row['Share Within Dimension']} of visible priority exposure in this dimension)"
            )

        st.markdown("### Executive Explanation")

        st.info(
            f"""
{priority} is not ranked highly because of a single metric alone. It appears as a top priority because modeled exposure, driver evidence, and organizational concentration point in the same direction.

The strongest visible concentration points are:

{chr(10).join(bullets)}

This means leadership should begin by validating whether these exposed workforce segments explain the observed pattern before considering any intervention.
            """
        )

def render_executive_intelligence_brief(
    action_df: pd.DataFrame,
    investigation_df: Optional[pd.DataFrame] = None,
    workforce_df: Optional[pd.DataFrame] = None,
) -> None:

    st.header("18. Executive Intelligence Brief")

    executive_brief_df, report = build_executive_intelligence_brief(
        action_df=action_df,
        investigation_df=investigation_df,
    )

    if report.errors:
        st.error("Executive Intelligence Brief errors:")
        for error in report.errors:
            st.write(f"- {error}")
        return

    if executive_brief_df.empty:
        st.info("No executive briefs could be generated.")
        return

    top = executive_brief_df.iloc[0]

    if report.warnings:
        with st.expander("Executive Brief Warnings", expanded=False):
            for warning in report.warnings:
                st.write(f"- {warning}")

    st.subheader("Board-Level Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        _card("Top Priority", top["workforce_priority"], "Highest-ranked investigation area")

    with c2:
        _card("Linked Exposure", _money(top["linked_modeled_exposure"]), "Modeled workforce exposure")

    with c3:
        _card("Evidence Strength", top["evidence_rating"], top["evidence_strength"])

    with c4:
        _card("Management Attention", top["attention_badge"], top["management_attention"])

    st.markdown("---")

    st.subheader("Executive Insight")

    st.markdown(
        f"""
<div style="
    background:#eef6ff;
    border-left:6px solid #2563eb;
    border-radius:16px;
    padding:24px;
    font-size:18px;
    line-height:1.65;
    color:#102a43;
">
    {top['executive_story']}
    <br><br>
    <strong>Board summary:</strong> {top['board_summary']}
</div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Top Workforce Priorities")
    _render_top_priority_chart(executive_brief_df)

    st.subheader("Executive Decision Matrix")

    matrix_cols = [
        "brief_rank",
        "workforce_priority",
        "recommended_investigation_area",
        "attention_badge",
        "evidence_rating",
        "linked_modeled_exposure",
        "top_department",
        "top_job_role",
        "top_job_level",
    ]

    matrix = executive_brief_df[matrix_cols].copy()
    matrix["linked_modeled_exposure"] = matrix["linked_modeled_exposure"].apply(_money)

    st.dataframe(matrix, use_container_width=True)

    selected_priority = st.selectbox(
        "Choose executive priority brief",
        executive_brief_df["workforce_priority"].tolist(),
    )

    selected = executive_brief_df[
        executive_brief_df["workforce_priority"] == selected_priority
    ].iloc[0]

    st.markdown("---")
    st.subheader(f"Priority #{int(selected['brief_rank'])}: {selected['workforce_priority']}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Linked Exposure", _money(selected["linked_modeled_exposure"]))
    c2.metric("Evidence", selected["evidence_strength"])
    c3.metric("Attention", selected["management_attention"])
    c4.metric("Evidence Drivers", int(selected["evidence_drivers"]))

    st.markdown(
        f"""
<div style="
    background:#ecfdf3;
    border-left:6px solid #16a34a;
    border-radius:16px;
    padding:22px;
    margin-top:12px;
    margin-bottom:20px;
    font-size:17px;
    line-height:1.6;
">
    <strong>Executive finding:</strong> {selected['executive_finding']}<br><br>
    <strong>Recommended investigation area:</strong> {selected['recommended_investigation_area']}<br>
    <strong>Primary management question:</strong> {selected['primary_management_question']}<br>
    <strong>Supporting variables:</strong> {selected['supporting_variables']}
</div>
        """,
        unsafe_allow_html=True,
    )

    _render_investigation_tree(selected)
    _render_executive_explainability(
        selected=selected,
        investigation_df=investigation_df,
    )
    st.subheader("Critical Executive Questions")

    for question in _split_pipe(selected["management_questions"]):
        st.write(f"- {question}")

    st.subheader("Recommended Investigation Focus")

    for action in _split_pipe(selected["review_actions"]):
        st.write(f"- {action}")

    st.subheader("Suggested Investigation Workflow")

    _render_step_flow(_split_pipe(selected["investigation_workflow"]))

    if investigation_df is not None and not investigation_df.empty:
        st.subheader("Visual Investigation Drilldown")

        available_dimensions = [
            dim for dim in [
                "Department",
                "Job Role",
                "Job Level",
                "Location",
                "Manager",
                "Business Unit",
                "Team",
            ]
            if not investigation_df[
                (investigation_df["driver_group"].astype(str) == str(selected_priority))
                & (investigation_df["dimension"].astype(str) == dim)
            ].empty
        ]

        if not available_dimensions:
            st.info("No drilldown dimensions are available for this priority.")
        else:
            tabs = st.tabs(available_dimensions)

            for tab, dimension in zip(tabs, available_dimensions):
                with tab:
                    _render_dimension_chart(
                        investigation_df=investigation_df,
                        selected_priority=selected_priority,
                        dimension=dimension,
                    )

                    subset = investigation_df[
                        (investigation_df["driver_group"].astype(str) == str(selected_priority))
                        & (investigation_df["dimension"].astype(str) == dimension)
                    ].copy()

                    display_cols = [
                        "segment",
                        "employees",
                        "avg_predicted_attrition_probability",
                        "total_segment_exposure",
                        "share_of_company_exposure",
                        "allocated_exposure_linked_to_priority",
                    ]

                    existing_cols = [c for c in display_cols if c in subset.columns]

                    subset = subset.sort_values(
                        "allocated_exposure_linked_to_priority",
                        ascending=False,
                    )

                    st.dataframe(subset[existing_cols], use_container_width=True)

    st.subheader("Executive Assessment")

    st.markdown(
        selected["executive_assessment"].replace("$", r"\$")
    )

    st.info(selected["limitations"])

    st.download_button(
        label="Download Executive Intelligence Brief",
        data=executive_brief_df.to_csv(index=False).encode("utf-8"),
        file_name="hcrl_executive_intelligence_brief.csv",
        mime="text/csv",
    )

    if investigation_df is not None and not investigation_df.empty:
        st.download_button(
            label="Download Investigation Drilldown",
            data=investigation_df.to_csv(index=False).encode("utf-8"),
            file_name="hcrl_executive_investigation_drilldown.csv",
            mime="text/csv",
        )

    with st.expander("Methodology & Enterprise Limitations"):
        st.write(
            """
HCRL Executive Intelligence Brief converts modeled workforce exposure,
driver evidence, and segment concentration into executive decision-support
narratives.

The brief identifies where leadership may begin investigation. It does not
establish causality, prescribe employment decisions, estimate ROI, guarantee
savings, or replace human management judgment.

Linked exposure is modeled exposure allocated to workforce priority areas
using prior Action Intelligence and Intervention Economics outputs.

Segment concentration identifies where the modeled exposure is most visible
across available organizational dimensions such as department, role, and job
level. Empty dimensions are hidden automatically.
            """
        )
