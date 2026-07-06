
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
        "workflow": ["Review promotion timelines", "Compare role tenure", "Assess internal mobility", "Review succession planning", "Prepare leadership findings"],
        "management_questions": ["Are employees remaining too long without role progression?", "Are promotion timelines consistent across departments and job levels?", "Are internal mobility paths visible and accessible?", "Are high-performing employees progressing internally?"],
        "review_actions": ["Review promotion timelines across exposed groups.", "Compare tenure and current-role duration by department and job level.", "Assess whether internal mobility is being used before external hiring.", "Review career ladders and succession planning for exposed roles."],
        "risk_if_ignored": ["Loss of experienced employees", "Weak internal mobility", "Higher external hiring dependency", "Leadership pipeline weakness"],
    },
    "Compensation": {
        "finding": "Compensation-related signals indicate a management review area for exposed workforce groups.",
        "primary_question": "Is compensation aligned with retention-sensitive roles and exposed workforce groups?",
        "workflow": ["Review compensation competitiveness", "Check pay progression", "Assess pay compression", "Review long-term incentives", "Prepare compensation findings"],
        "management_questions": ["Is compensation competitive for exposed roles?", "Are salary increases aligned with retention-sensitive groups?", "Are there signs of pay compression across job levels?", "Are long-term incentives being used effectively?"],
        "review_actions": ["Review compensation competitiveness for exposed roles.", "Compare pay progression by role, level, and department.", "Assess pay compression and salary increase patterns.", "Review long-term incentive structures for retention-sensitive groups."],
        "risk_if_ignored": ["Retention pressure in competitive roles", "Pay compression concerns", "Higher replacement cost exposure", "Lower perceived reward fairness"],
    },
    "Work Environment": {
        "finding": "Employee experience and work-environment signals show a visible management review area.",
        "primary_question": "Are employee experience issues concentrated in exposed teams, roles, or departments?",
        "workflow": ["Review employee experience signals", "Compare exposed teams", "Assess engagement and job satisfaction", "Review manager and team climate", "Prepare experience findings"],
        "management_questions": ["Are low satisfaction scores concentrated in specific teams?", "Do exposed groups report weaker job involvement?", "Are employee experience issues concentrated by manager or department?", "Are workload, recognition, and team climate being reviewed together?"],
        "review_actions": ["Review employee experience signals in exposed groups.", "Compare job satisfaction and involvement across departments.", "Assess team climate and recognition patterns.", "Review whether employee experience issues overlap with workload or manager signals."],
        "risk_if_ignored": ["Lower engagement", "Reduced team stability", "Higher voluntary turnover pressure", "Employee experience deterioration"],
    },
    "Manager Stability": {
        "finding": "Manager stability signals indicate a leadership-continuity review area.",
        "primary_question": "Are manager relationships or leadership continuity issues concentrated in exposed groups?",
        "workflow": ["Review manager continuity", "Compare teams by manager relationship duration", "Assess span of control", "Review leadership support", "Prepare management findings"],
        "management_questions": ["Do high-risk groups share unstable manager relationships?", "Are manager changes concentrated in exposed workforce areas?", "Do teams with shorter manager tenure show higher modeled risk?", "Should manager continuity be reviewed for critical roles?"],
        "review_actions": ["Review manager continuity in exposed workforce areas.", "Compare teams by manager relationship duration.", "Assess leadership support and span of control.", "Review manager coaching and team stability."],
        "risk_if_ignored": ["Leadership continuity risk", "Team disruption", "Lower trust in management", "Higher retention pressure in affected teams"],
    },
    "Workload": {
        "finding": "Workload signals indicate a capacity and staffing review area.",
        "primary_question": "Are workload demands concentrated in exposed workforce groups?",
        "workflow": ["Review workload indicators", "Compare overtime concentration", "Assess staffing levels", "Review scheduling or capacity gaps", "Prepare workload findings"],
        "management_questions": ["Is overtime concentrated in exposed roles or departments?", "Are workload spikes aligned with elevated modeled risk?", "Are staffing levels aligned with demand?", "Should scheduling or capacity planning be reviewed?"],
        "review_actions": ["Review overtime and workload patterns in exposed groups.", "Compare staffing levels against demand-sensitive roles.", "Assess scheduling and capacity planning.", "Review whether workload pressure overlaps with work-environment signals."],
        "risk_if_ignored": ["Burnout pressure", "Reduced productivity", "Higher absenteeism risk", "Workforce instability in overloaded groups"],
    },
    "Travel / Commute Burden": {
        "finding": "Travel and commute signals indicate a flexibility and location review area.",
        "primary_question": "Are commute, travel, or location requirements contributing to exposed workforce pressure?",
        "workflow": ["Review travel and commute requirements", "Compare exposed locations", "Assess flexibility options", "Review role-location alignment", "Prepare flexibility findings"],
        "management_questions": ["Are commute or travel expectations concentrated in exposed groups?", "Can flexibility reduce avoidable workforce friction?", "Are location requirements aligned with role needs?", "Do travel-heavy roles show higher modeled attrition risk?"],
        "review_actions": ["Review flexibility options for exposed groups.", "Compare travel burden and commute requirements across roles.", "Assess role-location alignment.", "Review location strategy and hybrid-work feasibility."],
        "risk_if_ignored": ["Avoidable friction for employees", "Reduced flexibility competitiveness", "Retention pressure in location-sensitive roles", "Higher dissatisfaction in travel-heavy work"],
    },
    "Training and Development": {
        "finding": "Training and development signals indicate a skills and mobility review area.",
        "primary_question": "Are development opportunities aligned with exposed workforce groups?",
        "workflow": ["Review training access", "Compare development participation", "Assess skills pathways", "Connect learning to internal mobility", "Prepare development findings"],
        "management_questions": ["Are development opportunities reaching exposed workforce groups?", "Are training investments aligned with retention-sensitive roles?", "Are learning pathways connected to career progression?", "Do exposed employees receive enough skill development?"],
        "review_actions": ["Review training access for exposed workforce groups.", "Compare training participation across departments and roles.", "Assess skills-development pathways.", "Connect training programs to internal mobility."],
        "risk_if_ignored": ["Skill stagnation", "Lower internal mobility", "Reduced readiness for future work", "Higher disengagement among growth-oriented employees"],
    },
    "Department": {
        "finding": "Department-level signals indicate that workforce exposure is concentrated in specific organizational areas.",
        "primary_question": "Which departments concentrate the largest modeled workforce exposure?",
        "workflow": ["Rank departments by exposure", "Compare local workforce conditions", "Review department practices", "Assess leadership differences", "Prepare department findings"],
        "management_questions": ["Which departments concentrate the largest exposure?", "Do departments differ in workforce conditions?", "Are local management practices contributing to risk variation?", "Should department leaders review local workforce conditions?"],
        "review_actions": ["Review department-level workforce exposure.", "Compare department conditions and manager patterns.", "Assess whether exposure is concentrated or broad-based.", "Begin investigation with the highest-exposure departments."],
        "risk_if_ignored": ["Localized workforce instability", "Uneven management practices", "Department-level retention pressure", "Hidden operational exposure"],
    },
    "Occupation": {
        "finding": "Occupation-level signals indicate role-specific workforce planning exposure.",
        "primary_question": "Which roles concentrate the largest modeled workforce exposure?",
        "workflow": ["Rank exposed roles", "Review labor-market sensitivity", "Assess hiring pipeline", "Review role-specific retention patterns", "Prepare occupation findings"],
        "management_questions": ["Are specific roles carrying disproportionate workforce exposure?", "Are role-level risks linked to labor-market pressure?", "Are hiring pipelines strong enough for exposed occupations?", "Should role-specific workforce planning be reviewed?"],
        "review_actions": ["Review role-specific workforce exposure.", "Assess hiring pipeline strength for exposed roles.", "Compare role-level retention patterns.", "Review workforce planning for critical occupations."],
        "risk_if_ignored": ["Role-level workforce shortages", "Replacement difficulty", "Operational dependency on exposed roles", "Higher hiring pipeline pressure"],
    },
    "Employee Experience": {
        "finding": "Employee experience signals indicate a lifecycle and retention-context review area.",
        "primary_question": "Are employee lifecycle patterns connected to exposed workforce groups?",
        "workflow": ["Review employee lifecycle patterns", "Compare tenure groups", "Assess progression and engagement", "Review experienced employee retention", "Prepare lifecycle findings"],
        "management_questions": ["Are experienced employees showing retention pressure?", "Are tenure patterns linked to career progression issues?", "Are long-tenured employees receiving enough development opportunities?", "Should employee lifecycle patterns be reviewed?"],
        "review_actions": ["Review lifecycle patterns in exposed groups.", "Compare tenure, progression, and experience signals.", "Assess whether long-tenured employees have growth pathways.", "Review employee experience by role and department."],
        "risk_if_ignored": ["Experienced employee loss", "Knowledge drain", "Weak lifecycle management", "Lower long-term workforce stability"],
    },
    "Education": {
        "finding": "Education signals should be interpreted as contextual workforce evidence.",
        "primary_question": "Are education patterns useful context for skills and workforce planning?",
        "workflow": ["Review education profile", "Compare skills pathways", "Assess development needs", "Avoid using education as a decision rule", "Prepare contextual findings"],
        "management_questions": ["Is education acting as a contextual workforce signal?", "Are skills-development pathways more useful than education categories?", "Do educational profiles differ across exposed groups?", "Should education be reviewed only as context?"],
        "review_actions": ["Review education only as workforce-planning context.", "Compare skills-development needs across exposed groups.", "Assess whether education signals connect to training pathways.", "Avoid treating education as a direct employment decision factor."],
        "risk_if_ignored": ["Weak skills planning", "Misread workforce capabilities", "Overreliance on credentials", "Missed development opportunities"],
    },
    "Performance": {
        "finding": "Performance signals indicate a review area for performance, progression, and retention alignment.",
        "primary_question": "Are performance systems aligned with progression and retention-sensitive groups?",
        "workflow": ["Review performance distribution", "Compare progression for high performers", "Assess reward alignment", "Review manager calibration", "Prepare performance findings"],
        "management_questions": ["Are performance ratings aligned with growth opportunities?", "Do performance processes support retention-sensitive employees?", "Are high-performing employees receiving progression opportunities?", "Should performance and career development be reviewed together?"],
        "review_actions": ["Review performance ratings in exposed workforce groups.", "Compare performance, compensation, and progression together.", "Assess manager calibration and promotion alignment.", "Review whether high-performing employees have clear growth paths."],
        "risk_if_ignored": ["High-performer disengagement", "Perceived unfairness", "Weak promotion alignment", "Reduced retention of strong contributors"],
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


def _split_pipe(value) -> List[str]:
    text = _clean(value)
    if text == "Not available":
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


def _domain_config(domain: str) -> Dict[str, object]:
    return DOMAIN_LANGUAGE.get(
        domain,
        {
            "finding": f"{domain} appears as a workforce investigation area.",
            "primary_question": f"What explains the workforce exposure associated with {domain}?",
            "workflow": ["Review exposed workforce data", "Compare segment concentration", "Interview relevant leaders", "Assess operating context", "Prepare leadership findings"],
            "management_questions": [f"Where is {domain} exposure concentrated?", f"Which workforce groups contribute most to this signal?", "Does the evidence align with management observations?", "What additional context is needed before action?"],
            "review_actions": [f"Review {domain} in exposed workforce groups.", "Compare concentration by department, role, and level.", "Validate findings with HR and business leadership.", "Prepare a management investigation summary."],
            "risk_if_ignored": ["Unresolved workforce exposure", "Delayed management attention", "Incomplete understanding of workforce risk", "Higher uncertainty in workforce planning"],
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


def _format_segment(value) -> str:
    text = _clean(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _render_step_flow(steps: List[str]) -> None:
    if not steps:
        return
    cols = st.columns(len(steps))
    for idx, step in enumerate(steps):
        with cols[idx]:
            st.markdown(
                f"""
                <div style="border:1px solid #d9dee7;border-radius:14px;padding:16px;min-height:105px;background:#ffffff;box-shadow:0 1px 2px rgba(0,0,0,0.04);">
                    <div style="font-size:13px;color:#667085;margin-bottom:6px;">Step {idx + 1}</div>
                    <div style="font-size:16px;font-weight:650;color:#1f2937;">{step}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


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

        priority_investigation = pd.DataFrame()
        if investigation_df is not None and not investigation_df.empty and "driver_group" in investigation_df.columns:
            priority_investigation = investigation_df[
                investigation_df["driver_group"].astype(str) == str(domain)
            ].copy()

        concentration_points: List[str] = []
        top_by_dimension: Dict[str, str] = {}

        if not priority_investigation.empty:
            priority_investigation["allocated_exposure_linked_to_priority"] = pd.to_numeric(
                priority_investigation["allocated_exposure_linked_to_priority"],
                errors="coerce",
            ).fillna(0)

            for dimension in ["Department", "Job Role", "Job Level", "Location", "Manager", "Business Unit", "Team", "Cost Center"]:
                dim_table = priority_investigation[
                    priority_investigation["dimension"].astype(str) == dimension
                ].copy()
                if dim_table.empty:
                    continue
                dim_table = dim_table.sort_values("allocated_exposure_linked_to_priority", ascending=False)
                top = dim_table.iloc[0]
                segment = _format_segment(top["segment"])
                exposure = _safe_float(top["allocated_exposure_linked_to_priority"])
                top_by_dimension[dimension] = segment
                concentration_points.append(f"{dimension}: {segment} ({_money(exposure)} allocated exposure)")

        top_department = top_by_dimension.get("Department", "Not available")
        top_role = top_by_dimension.get("Job Role", "Not available")
        top_level = top_by_dimension.get("Job Level", "Not available")

        linked_exposure = _safe_float(action["exposure_linked_to_intervention_area"])
        evidence_drivers = _safe_int(action["evidence_drivers"])
        evidence_strength = _clean(action["evidence_strength"])
        management_attention = _clean(action["management_attention"])

        concentration_sentence = "; ".join(concentration_points[:4]) if concentration_points else "segment concentration is not available from the uploaded data"

        executive_assessment = (
            f"{domain} represents Priority #{_safe_int(action['action_rank'])} because the uploaded workforce data shows "
            f"{evidence_drivers} supporting evidence signal(s) linked to approximately {_money(linked_exposure)} of modeled workforce exposure. "
            f"The most visible concentration points are {concentration_sentence}. "
            f"Leadership should begin investigation in these exposed workforce segments before expanding the review company-wide."
        )

        board_summary = (
            f"{domain} is the current workforce investigation priority. Linked exposure is {_money(linked_exposure)}, "
            f"evidence strength is {evidence_strength}, and management attention is {management_attention}."
        )

        rows.append(
            {
                "brief_rank": _safe_int(action["action_rank"]),
                "workforce_priority": domain,
                "executive_finding": config["finding"],
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
                "top_department": top_department,
                "top_job_role": top_role,
                "top_job_level": top_level,
                "concentration_points": " | ".join(concentration_points),
                "executive_assessment": executive_assessment,
                "board_summary": board_summary,
                "management_questions": " | ".join(config["management_questions"]),
                "review_actions": " | ".join(config["review_actions"]),
                "investigation_workflow": " | ".join(config["workflow"]),
                "risk_if_ignored": " | ".join(config["risk_if_ignored"]),
                "limitations": "This brief is evidence-aligned decision support. It does not prove causality, estimate ROI, guarantee savings, or prescribe automatic employment decisions.",
            }
        )

    brief_df = pd.DataFrame(rows).sort_values(["brief_rank", "linked_modeled_exposure"], ascending=[True, False]).reset_index(drop=True)

    warnings.append(
        "Executive Intelligence Brief synthesizes modeled exposure, driver evidence, and segment concentration into executive-ready narratives. It is not causal attribution, ROI estimation, or an automated employment decision."
    )

    return brief_df, ExecutiveBriefReport(len(brief_df), warnings, errors)


def render_executive_intelligence_brief(
    action_df: pd.DataFrame,
    investigation_df: Optional[pd.DataFrame] = None,
    workforce_df: Optional[pd.DataFrame] = None,
) -> None:

    st.header("18. Executive Intelligence Brief")

    executive_brief_df, report = build_executive_intelligence_brief(action_df=action_df, investigation_df=investigation_df)

    if report.errors:
        st.error("Executive Intelligence Brief errors:")
        for error in report.errors:
            st.write(f"- {error}")
        return

    if executive_brief_df.empty:
        st.info("No executive briefs could be generated.")
        return

    top = executive_brief_df.iloc[0]

    st.metric("Executive Briefs Generated", report.briefs_generated)

    if report.warnings:
        st.warning("Executive Intelligence Brief warnings:")
        for warning in report.warnings:
            st.write(f"- {warning}")

    st.subheader("Board-Level Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top Priority", top["workforce_priority"])
    c2.metric("Linked Exposure", _money(top["linked_modeled_exposure"]))
    c3.metric("Evidence Strength", top["evidence_strength"])
    c4.metric("Attention", top["management_attention"])

    st.success(top["board_summary"])

    st.subheader("Executive Assessment")
    st.markdown(
        f"""
        <div style="background:#eef6ff;border-left:6px solid #2563eb;border-radius:14px;padding:24px;margin-bottom:18px;font-size:18px;line-height:1.65;color:#102a43;">
            <strong>{top['workforce_priority']}</strong> represents the current highest-priority workforce investigation area.<br><br>
            {top['executive_assessment']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Where Leadership Should Start")
    c1, c2, c3 = st.columns(3)
    c1.metric("Top Department", top["top_department"])
    c2.metric("Top Job Role", top["top_job_role"])
    c3.metric("Top Job Level", top["top_job_level"])

    concentration_points = _split_pipe(top["concentration_points"])
    if concentration_points:
        st.markdown("**Primary concentration points:**")
        for point in concentration_points[:5]:
            st.write(f"- {point}")

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

    selected_priority = st.selectbox("Choose executive priority brief", executive_brief_df["workforce_priority"].tolist())
    selected = executive_brief_df[executive_brief_df["workforce_priority"] == selected_priority].iloc[0]

    st.subheader(f"Priority #{int(selected['brief_rank'])}: {selected['workforce_priority']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linked Exposure", _money(selected["linked_modeled_exposure"]))
    c2.metric("Evidence", selected["evidence_strength"])
    c3.metric("Attention", selected["management_attention"])
    c4.metric("Evidence Drivers", int(selected["evidence_drivers"]))

    st.markdown(
        f"""
        <div style="background:#ecfdf3;border-left:6px solid #16a34a;border-radius:14px;padding:22px;margin-top:12px;margin-bottom:20px;font-size:17px;line-height:1.6;">
            <strong>Executive finding:</strong> {selected['executive_finding']}<br><br>
            <strong>Recommended investigation area:</strong> {selected['recommended_investigation_area']}<br>
            <strong>Primary management question:</strong> {selected['primary_management_question']}<br>
            <strong>Supporting variables:</strong> {selected['supporting_variables']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Why This Matters")
    st.write(selected["executive_assessment"])

    st.subheader("Questions Leadership Should Answer")
    for question in _split_pipe(selected["management_questions"]):
        st.write(f"- {question}")

    st.subheader("Suggested Investigation Workflow")
    _render_step_flow(_split_pipe(selected["investigation_workflow"]))

    st.subheader("Management Review Areas")
    for action in _split_pipe(selected["review_actions"]):
        st.write(f"- {action}")

    st.subheader("Potential Business Risks if Ignored")
    for risk in _split_pipe(selected["risk_if_ignored"]):
        st.write(f"- {risk}")

    selected_points = _split_pipe(selected["concentration_points"])
    if selected_points:
        st.subheader("Primary Exposure Concentration")
        for point in selected_points[:6]:
            st.write(f"- {point}")

    if investigation_df is not None and not investigation_df.empty and "driver_group" in investigation_df.columns:
        selected_investigation = investigation_df[
            investigation_df["driver_group"].astype(str) == str(selected["workforce_priority"])
        ].copy()
        if not selected_investigation.empty:
            st.subheader("Investigation Drilldown")
            for dimension in selected_investigation["dimension"].dropna().unique():
                dim_table = selected_investigation[selected_investigation["dimension"] == dimension].copy()
                if dim_table.empty:
                    continue
                dim_table = dim_table.sort_values("allocated_exposure_linked_to_priority", ascending=False)
                display_cols = [
                    "segment",
                    "employees",
                    "avg_predicted_attrition_probability",
                    "total_segment_exposure",
                    "share_of_company_exposure",
                    "allocated_exposure_linked_to_priority",
                ]
                existing_display_cols = [col for col in display_cols if col in dim_table.columns]
                with st.expander(f"Exposure concentration by {dimension}", expanded=False):
                    st.dataframe(dim_table[existing_display_cols], use_container_width=True)

    st.subheader("Full Executive Brief Table")
    full_cols = [
        "brief_rank",
        "workforce_priority",
        "recommended_investigation_area",
        "actionability",
        "evidence_strength",
        "management_attention",
        "linked_modeled_exposure",
        "evidence_drivers",
        "top_department",
        "top_job_role",
        "top_job_level",
        "executive_assessment",
    ]
    st.dataframe(executive_brief_df[full_cols], use_container_width=True)

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

