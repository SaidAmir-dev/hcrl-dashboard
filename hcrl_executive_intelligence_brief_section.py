from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px


@dataclass
class ExecutiveBriefReport:
    briefs_generated: int
    warnings: List[str]
    errors: List[str]


# =====================================================
# DOMAIN LANGUAGE
# =====================================================

DOMAIN_LANGUAGE: Dict[str, Dict[str, object]] = {
    "Career Progression": {
        "finding": "Promotion-related workforce signals represent the largest visible management investigation area.",
        "executive_label": "Career progression and internal mobility",
        "primary_question": "Are promotion and internal mobility pathways aligned with exposed workforce groups?",
        "workflow": [
            "Validate promotion evidence",
            "Review exposed departments",
            "Compare tenure and role duration",
            "Assess internal mobility paths",
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
        "risk_if_ignored": [
            "Loss of experienced employees.",
            "Weak internal mobility.",
            "Higher external hiring dependency.",
            "Leadership pipeline weakness.",
        ],
    },
    "Compensation": {
        "finding": "Compensation-related signals indicate a management review area for exposed workforce groups.",
        "executive_label": "Compensation competitiveness and pay progression",
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
        "risk_if_ignored": [
            "Retention pressure in competitive roles.",
            "Pay compression concerns.",
            "Higher replacement-cost exposure.",
            "Lower perceived reward fairness.",
        ],
    },
    "Work Environment": {
        "finding": "Employee experience and work-environment signals show a visible management review area.",
        "executive_label": "Employee experience and work climate",
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
        "risk_if_ignored": [
            "Lower engagement.",
            "Reduced team stability.",
            "Higher voluntary turnover pressure.",
            "Employee experience deterioration.",
        ],
    },
    "Manager Stability": {
        "finding": "Manager stability signals indicate a leadership-continuity review area.",
        "executive_label": "Manager continuity and leadership stability",
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
        "risk_if_ignored": [
            "Leadership continuity risk.",
            "Team disruption.",
            "Lower trust in management.",
            "Higher retention pressure in affected teams.",
        ],
    },
    "Workload": {
        "finding": "Workload signals indicate a capacity and staffing review area.",
        "executive_label": "Workload, overtime, and capacity pressure",
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
        "risk_if_ignored": [
            "Burnout pressure.",
            "Capacity mismatch.",
            "Reduced productivity stability.",
            "Higher dissatisfaction in workload-heavy roles.",
        ],
    },
    "Travel / Commute Burden": {
        "finding": "Travel and commute signals indicate a flexibility and location review area.",
        "executive_label": "Travel burden, commute, and flexibility",
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
        "risk_if_ignored": [
            "Avoidable employee friction.",
            "Reduced flexibility competitiveness.",
            "Retention pressure in location-sensitive roles.",
            "Higher dissatisfaction in travel-heavy work.",
        ],
    },
    "Training and Development": {
        "finding": "Training and development signals indicate a skills and mobility review area.",
        "executive_label": "Training, development, and internal mobility readiness",
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
        "risk_if_ignored": [
            "Skill stagnation.",
            "Lower internal mobility.",
            "Reduced readiness for future work.",
            "Higher disengagement among growth-oriented employees.",
        ],
    },
}


# =====================================================
# HELPERS
# =====================================================

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


def _pct(value) -> str:
    return f"{_safe_float(value) * 100:,.1f}%"


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
    return text[:-2] if text.endswith(".0") else text


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
            "executive_label": domain,
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
            "risk_if_ignored": [
                "Unresolved workforce exposure.",
                "Delayed management review.",
                "Incomplete workforce risk understanding.",
                "Higher uncertainty in workforce planning.",
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


def _section_card(title: str, body: str, border_color: str = "#2563eb", bg: str = "#eef6ff") -> None:
    st.markdown(
        f"""
<div style="
    background:{bg};
    border-left:6px solid {border_color};
    border-radius:16px;
    padding:24px;
    font-size:17px;
    line-height:1.65;
    color:#102a43;
">
    <strong>{title}</strong><br><br>
    {body}
</div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# CORE BUILD
# =====================================================

def _investigation_points(
    investigation_df: pd.DataFrame,
    domain: str,
) -> Tuple[List[str], Dict[str, str], Dict[str, float]]:

    points: List[str] = []
    top_by_dimension: Dict[str, str] = {}
    exposure_by_dimension: Dict[str, float] = {}

    if investigation_df is None or investigation_df.empty:
        return points, top_by_dimension, exposure_by_dimension

    required = [
        "driver_group",
        "dimension",
        "segment",
        "allocated_exposure_linked_to_priority",
    ]

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
        dim_df = df[df["dimension"].astype(str) == dimension].copy()
        if dim_df.empty:
            continue

        dim_df = dim_df.sort_values(
            "allocated_exposure_linked_to_priority",
            ascending=False,
        )

        top = dim_df.iloc[0]
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

        points, top_dim, exp_dim = _investigation_points(investigation_df, domain)

        linked_exposure = _safe_float(action["exposure_linked_to_intervention_area"])
        evidence_drivers = _safe_int(action["evidence_drivers"])
        evidence_strength = _clean(action["evidence_strength"])
        management_attention = _clean(action["management_attention"])

        concentration_sentence = (
            "; ".join(points[:4])
            if points
            else "segment concentration is not available from the uploaded data"
        )

        top_department = top_dim.get("Department", "Not available")
        top_role = top_dim.get("Job Role", "Not available")
        top_level = top_dim.get("Job Level", "Not available")

        executive_story = (
            f"{domain} emerged as a visible workforce investigation priority, "
            f"representing approximately {_money(linked_exposure)} of modeled exposure. "
            f"The strongest available concentration appears in {top_department}, "
            f"with role-level concentration around {top_role}. "
            f"Leadership should validate whether this pattern reflects the workforce issue described by: "
            f"{config['primary_question']}"
        )

        executive_assessment = (
            f"{domain} represents Priority #{_safe_int(action['action_rank'])}. "
            f"The uploaded workforce data shows {evidence_drivers} supporting evidence signal(s) "
            f"linked to approximately {_money(linked_exposure)} of modeled workforce exposure. "
            f"The most visible concentration points are {concentration_sentence}. "
            f"Leadership should begin investigation in these exposed workforce segments before expanding the review company-wide. "
            f"This is an investigation priority, not a causal claim or prescribed intervention."
        )

        rows.append(
            {
                "brief_rank": _safe_int(action["action_rank"]),
                "workforce_priority": domain,
                "executive_label": config["executive_label"],
                "recommended_investigation_area": _clean(action["intervention_area"]),
                "executive_finding": config["finding"],
                "primary_management_question": config["primary_question"],
                "actionability": _clean(action["actionability"]),
                "evidence_strength": evidence_strength,
                "evidence_rating": _star_rating(evidence_strength),
                "management_attention": management_attention,
                "attention_badge": _attention_badge(management_attention),
                "linked_modeled_exposure": linked_exposure,
                "evidence_drivers": evidence_drivers,
                "supporting_variables": _clean(action["supporting_variables"]),
                "top_department": top_department,
                "top_department_exposure": exp_dim.get("Department", 0.0),
                "top_job_role": top_role,
                "top_job_role_exposure": exp_dim.get("Job Role", 0.0),
                "top_job_level": top_level,
                "top_job_level_exposure": exp_dim.get("Job Level", 0.0),
                "top_location": top_dim.get("Location", "Not available"),
                "concentration_points": " | ".join(points),
                "executive_story": executive_story,
                "executive_assessment": executive_assessment,
                "board_summary": (
                    f"{domain}: {_money(linked_exposure)} linked exposure, "
                    f"{evidence_strength} evidence strength, {management_attention}."
                ),
                "management_questions": " | ".join(config["questions"]),
                "review_actions": " | ".join(config["focus"]),
                "investigation_workflow": " | ".join(config["workflow"]),
                "risk_if_ignored": " | ".join(config["risk_if_ignored"]),
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


# =====================================================
# EXECUTIVE INTELLIGENCE DERIVED INSIGHTS
# =====================================================

def _build_cross_priority_insights(executive_brief_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dimension, label_col in [
        ("Department", "top_department"),
        ("Job Role", "top_job_role"),
        ("Job Level", "top_job_level"),
        ("Location", "top_location"),
    ]:
        if label_col not in executive_brief_df.columns:
            continue

        temp = executive_brief_df[
            executive_brief_df[label_col].notna()
            & (executive_brief_df[label_col].astype(str) != "Not available")
        ].copy()

        if temp.empty:
            continue

        grouped = (
            temp.groupby(label_col)
            .agg(
                priority_count=("workforce_priority", "nunique"),
                linked_exposure=("linked_modeled_exposure", "sum"),
                priorities=("workforce_priority", lambda x: " | ".join(sorted(set(map(str, x))))),
            )
            .reset_index()
            .rename(columns={label_col: "segment"})
        )

        grouped["dimension"] = dimension
        rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["priority_count", "linked_exposure"], ascending=[False, False])
    return out


def _render_cross_priority_intelligence(executive_brief_df: pd.DataFrame) -> None:
    insights = _build_cross_priority_insights(executive_brief_df)

    st.subheader("Cross-Priority Intelligence")

    if insights.empty:
        st.info("No recurring cross-priority concentration was detected from available drilldown dimensions.")
        return

    top = insights.iloc[0]

    _section_card(
        "Recurring Organizational Theme",
        (
            f"The strongest recurring concentration is <strong>{top['segment']}</strong> "
            f"within <strong>{top['dimension']}</strong>. It appears across "
            f"<strong>{int(top['priority_count'])}</strong> workforce priority area(s), "
            f"with combined linked exposure of <strong>{_money(top['linked_exposure'])}</strong>."
        ),
        border_color="#7c3aed",
        bg="#f5f3ff",
    )

    display = insights.head(10).copy()
    display["linked_exposure"] = display["linked_exposure"].apply(_money)

    st.dataframe(
        display[
            [
                "dimension",
                "segment",
                "priority_count",
                "linked_exposure",
                "priorities",
            ]
        ].rename(
            columns={
                "dimension": "Dimension",
                "segment": "Segment",
                "priority_count": "Priority Count",
                "linked_exposure": "Combined Linked Exposure",
                "priorities": "Related Priorities",
            }
        ),
        use_container_width=True,
    )


def _render_executive_recommendations(selected: pd.Series) -> None:
    st.subheader("Top Executive Recommendations")

    focus_items = _split_pipe(selected["review_actions"])
    questions = _split_pipe(selected["management_questions"])

    recommendations: List[str] = []

    if focus_items:
        recommendations.extend(focus_items[:3])

    if questions:
        recommendations.append(f"Use leadership review to answer: {questions[0]}")

    recommendations.append(
        "Validate the pattern with HR and business leaders before designing any intervention."
    )

    for i, item in enumerate(recommendations[:5], start=1):
        st.markdown(
            f"""
<div style="
    border:1px solid #e5e7eb;
    border-radius:14px;
    padding:16px;
    margin-bottom:10px;
    background:#ffffff;
">
    <div style="font-size:13px;color:#667085;">Recommendation {i}</div>
    <div style="font-size:16px;font-weight:650;color:#111827;">{item}</div>
</div>
            """,
            unsafe_allow_html=True,
        )


# =====================================================
# RENDER HELPERS
# =====================================================

def _render_priority_chart(executive_brief_df: pd.DataFrame) -> None:
    chart_df = executive_brief_df.sort_values(
        "linked_modeled_exposure",
        ascending=False,
    ).head(8)

    fig = px.bar(
        chart_df,
        x="workforce_priority",
        y="linked_modeled_exposure",
        labels={
            "workforce_priority": "Workforce Priority",
            "linked_modeled_exposure": "Linked Modeled Exposure",
        },
        text="linked_modeled_exposure",
        hover_data=["evidence_strength", "management_attention"],
    )

    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig.update_layout(height=430, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _render_dimension_chart(
    investigation_df: pd.DataFrame,
    selected_priority: str,
    dimension: str,
) -> None:

    df = investigation_df[
        (investigation_df["driver_group"].astype(str) == str(selected_priority))
        & (investigation_df["dimension"].astype(str) == dimension)
    ].copy()

    if df.empty:
        st.info(f"No {dimension} drilldown available.")
        return

    df["allocated_exposure_linked_to_priority"] = pd.to_numeric(
        df["allocated_exposure_linked_to_priority"],
        errors="coerce",
    ).fillna(0)

    df = df.sort_values(
        "allocated_exposure_linked_to_priority",
        ascending=False,
    ).head(10)

    fig = px.bar(
        df,
        x="segment",
        y="allocated_exposure_linked_to_priority",
        hover_data=[
            col for col in [
                "employees",
                "avg_predicted_attrition_probability",
                "total_segment_exposure",
                "share_of_company_exposure",
            ]
            if col in df.columns
        ],
        labels={
            "segment": dimension,
            "allocated_exposure_linked_to_priority": "Allocated Exposure",
        },
    )

    fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside")
    fig.update_layout(height=430, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    display_cols = [
        "segment",
        "employees",
        "avg_predicted_attrition_probability",
        "total_segment_exposure",
        "share_of_company_exposure",
        "allocated_exposure_linked_to_priority",
    ]

    existing_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[existing_cols], use_container_width=True)


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


def _render_investigation_path(selected: pd.Series) -> None:
    nodes = [
        selected["workforce_priority"],
        selected["top_department"],
        selected["top_job_role"],
        f"Job Level {selected['top_job_level']}",
        _money(selected["linked_modeled_exposure"]),
    ]

    nodes = [x for x in nodes if _clean(x) != "Not available"]

    st.subheader("Highest Exposure Across Organizational Dimensions")

    html = "<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>"

    for i, node in enumerate(nodes):
        html += f"""
<div style="
    border:1px solid #d1d5db;
    border-radius:999px;
    padding:10px 16px;
    background:#ffffff;
    font-weight:650;
    color:#111827;
">
    {node}
</div>
        """
        if i < len(nodes) - 1:
            html += "<div style='font-size:22px;color:#667085;'>→</div>"

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _render_executive_assessment_card(selected: pd.Series) -> None:
    dept = selected.get("top_department", "Not available")
    role = selected.get("top_job_role", "Not available")
    level = selected.get("top_job_level", "Not available")

    body = (
        f"<strong>{selected['workforce_priority']}</strong> is ranked as "
        f"<strong>Priority #{int(selected['brief_rank'])}</strong>, with "
        f"<strong>{_money(selected['linked_modeled_exposure'])}</strong> of linked modeled exposure. "
        f"The largest visible concentration appears in <strong>{dept}</strong>, "
        f"with role concentration around <strong>{role}</strong> and job-level concentration around "
        f"<strong>{level}</strong>. The finding is supported by "
        f"<strong>{int(selected['evidence_drivers'])}</strong> evidence driver(s): "
        f"<strong>{selected['supporting_variables']}</strong>."
    )

    _section_card(
        "Executive Assessment",
        body,
        border_color="#0f766e",
        bg="#ecfdf5",
    )


# =====================================================
# MAIN RENDER FUNCTION
# =====================================================

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

    # Developer-style warnings are intentionally hidden from the top of the report.
    # They remain represented in methodology/governance text at the bottom.

    total_exposure = executive_brief_df["linked_modeled_exposure"].sum()
    priority_areas = len(executive_brief_df)
    immediate_reviews = executive_brief_df[
        executive_brief_df["management_attention"].astype(str) == "Immediate Management Review"
    ].shape[0]
    total_evidence_drivers = executive_brief_df["evidence_drivers"].sum()
    visible_segments = (
        executive_brief_df[["top_department", "top_job_role", "top_job_level"]]
        .replace("Not available", pd.NA)
        .stack()
        .dropna()
        .nunique()
    )

    st.subheader("Executive Workforce Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        _card("Enterprise Exposure", _money(total_exposure), "Total linked modeled exposure")
    with c2:
        _card("Priority Areas", str(priority_areas), "Identified workforce domains")
    with c3:
        _card("Immediate Reviews", str(immediate_reviews), "Highest attention areas")
    with c4:
        _card("Evidence Drivers", str(int(total_evidence_drivers)), "Supporting evidence signals")
    with c5:
        _card("Visible Segments", str(int(visible_segments)), "Departments, roles, and levels")

    st.markdown("---")

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

    st.subheader("Executive Insight")

    insight_body = (
        f"<strong>{top['workforce_priority']}</strong> is the highest-ranked workforce investigation priority. "
        f"It represents <strong>{_money(top['linked_modeled_exposure'])}</strong> of linked modeled exposure, "
        f"with the strongest visible concentration in <strong>{top['top_department']}</strong> and "
        f"<strong>{top['top_job_role']}</strong>. Leadership should begin by validating this pattern "
        f"before expanding the review to lower-ranked workforce priorities."
    )

    _section_card("Executive Insight", insight_body, border_color="#2563eb", bg="#eef6ff")

    st.subheader("Enterprise Workforce Priorities")
    _render_priority_chart(executive_brief_df)

    st.subheader("Executive Decision Matrix")

    matrix = executive_brief_df[
        [
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
    ].copy()

    matrix["linked_modeled_exposure"] = matrix["linked_modeled_exposure"].apply(_money)

    matrix = matrix.rename(
        columns={
            "brief_rank": "Priority",
            "workforce_priority": "Workforce Priority",
            "recommended_investigation_area": "Investigation Area",
            "attention_badge": "Attention",
            "evidence_rating": "Evidence",
            "linked_modeled_exposure": "Linked Exposure",
            "top_department": "Top Department",
            "top_job_role": "Top Job Role",
            "top_job_level": "Top Job Level",
        }
    )

    st.dataframe(matrix, use_container_width=True)

    _render_cross_priority_intelligence(executive_brief_df)

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

    _render_executive_assessment_card(selected)

    _render_investigation_path(selected)

    st.subheader("Critical Executive Questions")
    for question in _split_pipe(selected["management_questions"]):
        st.write(f"- {question}")

    st.subheader("Recommended Investigation Focus")
    for action in _split_pipe(selected["review_actions"]):
        st.write(f"- {action}")

    _render_executive_recommendations(selected)

    st.subheader("Suggested Investigation Workflow")
    _render_step_flow(_split_pipe(selected["investigation_workflow"]))

    st.subheader("Potential Business Risks if Ignored")
    for risk in _split_pipe(selected["risk_if_ignored"]):
        st.write(f"- {risk}")

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
                "Cost Center",
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


