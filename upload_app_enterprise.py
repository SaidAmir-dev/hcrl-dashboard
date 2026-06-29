"""HCRL Enterprise App."""

import streamlit as st
import pandas as pd

from hcrl_schema import standardize_workforce_data
from hcrl_onet_mapper import map_to_onet

from hcrl_driver_recommendation_engine import build_driver_recommendations
from hcrl_attrition_driver_engine import build_attrition_driver_table

from hcrl_task_intelligence_engine import attach_task_intelligence
from hcrl_narrative_engine import build_workforce_narratives
from hcrl_workforce_action_investigation_engine import (
    build_workforce_action_investigation,
)
from hcrl_intervention_intelligence_engine import (
    build_intervention_intelligence_table,
)

from hcrl_intervention_economics_engine import (
    build_intervention_economics_table,
)
from hcrl_risk_engine import estimate_attrition_risk
from hcrl_workforce_opportunity_engine import (
    build_workforce_opportunity_table,
)
from hcrl_segmentation_engine import build_segmentation_table
from hcrl_cost_engine import estimate_expected_attrition_cost
from hcrl_ai_readiness_engine import attach_ai_readiness
from hcrl_prioritization_engine import build_prioritization_table
from hcrl_workforce_leverage_engine import (
    build_workforce_leverage_table,
)
from hcrl_scenario_engine import build_scenario_table
from hcrl_executive_report_engine import (
    build_executive_focus_table
)
from hcrl_decision_engine import build_segment_decision_table
from hcrl_action_intelligence_engine import (
    build_action_intelligence_table
)

st.set_page_config(
    page_title="HCRL Enterprise Workforce Intelligence",
    layout="wide",
)

st.title("Human Capital Risk Lab")
st.subheader("Enterprise Workforce Transformation Intelligence Platform")

st.write(
    "HCRL analyzes workforce risk, economic exposure, occupation-task structure, "
    "O*NET intelligence, and intervention economics. The platform does not make "
    "firing recommendations."
)

st.divider()

st.header("1. Upload Workforce Data")

workforce_file = st.file_uploader("Upload company workforce CSV", type=["csv"])

if workforce_file is None:
    st.info("Upload a company workforce file to begin.")
    st.stop()

raw_df = pd.read_csv(workforce_file)

st.success("Workforce file uploaded successfully.")
st.write(f"Rows uploaded: {len(raw_df):,}")
st.write(f"Columns uploaded: {len(raw_df.columns):,}")

with st.expander("Preview uploaded data"):
    st.dataframe(raw_df.head(20), use_container_width=True)


st.header("2. HCRL Schema Validation")

df, schema_report = standardize_workforce_data(raw_df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Source Type", schema_report.source_type)
c2.metric("Mapped Fields", len(schema_report.mapped_columns))
c3.metric("Model Features", len(schema_report.model_feature_columns))
c4.metric("Unmapped Columns", len(schema_report.unmapped_columns))

st.subheader("Mapped Columns")

mapped_columns_df = pd.DataFrame(
    [{"HCRL Field": k, "Source Column": v} for k, v in schema_report.mapped_columns.items()]
)

if mapped_columns_df.empty:
    st.info("No canonical HCRL fields were automatically mapped.")
else:
    st.dataframe(mapped_columns_df, use_container_width=True)

if schema_report.warnings:
    st.warning("Schema warnings:")
    for warning in schema_report.warnings:
        st.write(f"- {warning}")

if schema_report.errors:
    st.error("Schema errors:")
    for error in schema_report.errors:
        st.write(f"- {error}")
    st.stop()


st.header("3. O*NET Occupation Intelligence")

try:
    onet_reference = pd.read_csv("onet_occupation_feature_table.csv")
    df, onet_report = map_to_onet(df, onet_reference)

    total_rows = len(df)
    accepted_coverage = onet_report.accepted_matches / total_rows if total_rows > 0 else 0
    review_share = onet_report.review_required_matches / total_rows if total_rows > 0 else 0

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Role Column", str(onet_report.role_column))
    o2.metric("Accepted Coverage", f"{accepted_coverage:.1%}")
    o3.metric("Accepted Matches", f"{onet_report.accepted_matches:,}")
    o4.metric("Review Queue", f"{onet_report.review_required_matches:,} ({review_share:.1%})")

    st.write(
        "Accepted coverage includes only deterministic matches. "
        "Review queue contains candidate mappings that require human validation before enterprise reporting."
    )

    debug_cols = [
        "employee_id",
        "job_title",
        "department",
        "normalized_title",
        "title_function",
        "title_level",
        "title_normalization_method",
        "candidate_titles",
        "matched_onet_title",
        "matched_onet_code",
        "onet_match_score",
        "onet_match_method",
        "onet_match_status",
    ]

    available_debug_cols = [c for c in debug_cols if c in df.columns]

    st.subheader("O*NET Mapping Debug Table")
    st.dataframe(df[available_debug_cols].head(150), use_container_width=True)

except Exception as e:
    st.error(f"O*NET mapping failed: {e}")


st.header("4. Task Intelligence")

try:
    df, role_task_table, task_report = attach_task_intelligence(df)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Task Summary Available", "Yes" if task_report.task_summary_available else "No")
    t2.metric("Task Portfolio Available", "Yes" if task_report.task_portfolio_available else "No")
    t3.metric("Matched Occupations", f"{task_report.matched_occupations:,}")
    t4.metric("Unmatched Occupations", f"{task_report.unmatched_occupations:,}")

    if task_report.warnings:
        st.warning("Task Intelligence warnings:")
        for warning in task_report.warnings:
            st.write(f"- {warning}")

    if task_report.errors:
        st.error("Task Intelligence errors:")
        for error in task_report.errors:
            st.write(f"- {error}")

    if not role_task_table.empty:
        st.subheader("Occupation Task Intelligence")
        st.dataframe(role_task_table, use_container_width=True)

        st.download_button(
            label="Download Occupation Task Table",
            data=role_task_table.to_csv(index=False).encode("utf-8"),
            file_name="hcrl_occupation_task_table.csv",
            mime="text/csv",
        )
    else:
        st.info("Task table unavailable. This usually means O*NET mapping failed or task portfolio files are missing.")

except Exception as e:
    st.error(f"Task Intelligence Engine failed: {e}")


st.header("5. Attrition Risk Engine")

df, risk_report = estimate_attrition_risk(df, schema_report.model_feature_columns)

r1, r2, r3, r4 = st.columns(4)
r1.metric("Risk Source", risk_report.risk_source)
r2.metric("Model Used", str(risk_report.model_used))
r3.metric("Observations", f"{risk_report.n_observations:,}")
r4.metric("Features", f"{risk_report.n_features:,}")

if risk_report.warnings:
    st.warning("Risk engine warnings:")
    for warning in risk_report.warnings:
        st.write(f"- {warning}")

if risk_report.errors:
    st.error("Risk engine errors:")
    for error in risk_report.errors:
        st.write(f"- {error}")

if "predicted_attrition_probability" in df.columns:
    valid_risk = pd.to_numeric(df["predicted_attrition_probability"], errors="coerce").dropna()
    if not valid_risk.empty:
        st.metric("Average Predicted Attrition Probability", f"{valid_risk.mean():.1%}")
    else:
        st.info("Attrition probabilities are unavailable for this dataset.")


st.header("6. Economic Exposure Engine")

df, cost_report = estimate_expected_attrition_cost(df)

e1, e2 = st.columns(2)
e1.metric("Cost Source", cost_report.cost_source)
e2.metric("Rows Analyzed", f"{cost_report.n_observations:,}")

if cost_report.warnings:
    st.warning("Cost engine warnings:")
    for warning in cost_report.warnings:
        st.write(f"- {warning}")

if cost_report.errors:
    st.error("Cost engine errors:")
    for error in cost_report.errors:
        st.write(f"- {error}")

if "expected_attrition_cost" in df.columns:
    valid_cost = pd.to_numeric(df["expected_attrition_cost"], errors="coerce").dropna()
    if not valid_cost.empty:
        st.metric("Total Expected Attrition Cost", f"${valid_cost.sum():,.0f}")
    else:
        st.info("Expected attrition cost is unavailable until risk and replacement-cost inputs exist.")

# =====================================================
# 7. AI READINESS ENGINE
# =====================================================

st.header("7. AI Readiness Engine")

df, ai_report = attach_ai_readiness(
    df,
    "onet_occupation_feature_table.csv"
)

if ai_report.errors:
    st.error("AI readiness errors:")
    for error in ai_report.errors:
        st.write(error)

a1, a2, a3 = st.columns(3)

with a1:
    st.metric(
        "Matched Workforce Rows",
        ai_report.matched_rows
    )

with a2:
    st.metric(
        "Unmatched Workforce Rows",
        ai_report.unmatched_rows
    )

with a3:
    st.metric(
        "AI Dimensions",
        len(ai_report.dimension_columns_used)
    )

if ai_report.warnings:
    st.warning("AI readiness warnings:")
    for warning in ai_report.warnings:
        st.write(f"- {warning}")

available_cols = [
    col
    for col in [
        "ai_digital_work_percentile",
        "ai_analytical_cognitive_work_percentile",
        "ai_human_interaction_work_percentile",
        "ai_physical_manual_work_percentile",
    ]
    if col in df.columns
]

if available_cols:
    st.subheader("AI Readiness Preview")

    preview_cols = []

    if "job_title" in df.columns:
        preview_cols.append("job_title")

    if "matched_onet_title" in df.columns:
        preview_cols.append("matched_onet_title")

    preview_cols += available_cols

    st.dataframe(
        df[preview_cols].head(50),
        use_container_width=True
    )

# =====================================================
# 8. WORKFORCE PRIORITIZATION
# =====================================================

st.header("8. Workforce Prioritization")

priority_table, priority_report = (
    build_prioritization_table(df)
)

if priority_report.errors:
    st.error("Prioritization errors:")
    for error in priority_report.errors:
        st.write(error)

else:

    st.metric(
        "Occupations Ranked",
        priority_report.occupations_analyzed
    )

    st.dataframe(
        priority_table,
        use_container_width=True
    )

    st.download_button(
        label="Download Workforce Prioritization Table",
        data=priority_table.to_csv(index=False).encode("utf-8"),
        file_name="hcrl_workforce_prioritization.csv",
        mime="text/csv",
    )

# =====================================================
# 9. EXECUTIVE REPORT
# =====================================================

st.header("9. Executive Management Report")

executive_table, executive_report = (
    build_executive_focus_table(
        priority_table
    )
)

if executive_report.errors:

    st.error(
        "Executive report errors:"
    )

    for error in executive_report.errors:
        st.write(error)

else:

    e1, e2, e3, e4 = st.columns(4)

    e1.metric(
        "Expected Attrition Cost",
        f"${executive_report.total_expected_attrition_cost:,.0f}"
    )

    e2.metric(
        "Top Cost Driver",
        executive_report.top_cost_driver
    )

    e3.metric(
        "Top Cost Share",
        f"{executive_report.top_cost_share:.1f}%"
    )

    e4.metric(
        "Focus Areas",
        executive_report.focus_areas_identified
    )

    st.subheader(
        "Management Focus Areas"
    )

    st.dataframe(
        executive_table,
        use_container_width=True,
    )

    st.download_button(
        label="Download Executive Report Table",
        data=executive_table.to_csv(index=False).encode("utf-8"),
        file_name="hcrl_executive_report.csv",
        mime="text/csv",
    )

st.header("10. Decision Intelligence")

segment_options = [
    col for col in [
        "job_title",
        "department",
        "location",
        "matched_onet_title",
        "occupation_code",
        "matched_onet_code",
    ]
    if col in df.columns
]

if not segment_options:
    st.warning("No usable segment column found.")
else:
    segment_col = st.selectbox("Choose workforce segmentation variable", segment_options)

    decision_table, decision_report = build_segment_decision_table(df, segment_col=segment_col)

    if decision_report.warnings:
        st.warning("Decision engine warnings:")
        for warning in decision_report.warnings:
            st.write(f"- {warning}")

    if decision_report.errors:
        st.error("Decision engine errors:")
        for error in decision_report.errors:
            st.write(f"- {error}")
    else:
        st.metric("Segments Analyzed", f"{decision_report.n_segments:,}")
        st.dataframe(decision_table, use_container_width=True)

        st.download_button(
            label="Download Segment Decision Table",
            data=decision_table.to_csv(index=False).encode("utf-8"),
            file_name="hcrl_segment_decision_table.csv",
            mime="text/csv",
        )


st.header("11. Download Full HCRL Output")

st.download_button(
    label="Download Full HCRL Analyzed Dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="hcrl_enterprise_analyzed_workforce.csv",
    mime="text/csv",
)

# =====================================================
# 12. WORKFORCE SEGMENTATION INTELLIGENCE
# =====================================================

st.header("12. Workforce Segmentation Intelligence")

segmentation_options = [
    col for col in [
        "department",
        "location",
        "job_title",
        "matched_onet_title",
        "business_unit",
        "country",
        "salary_band",
    ]
    if col in df.columns
]

if not segmentation_options:
    st.warning("No usable segmentation columns found.")

else:
    selected_segment = st.selectbox(
        "Choose organizational segment",
        segmentation_options,
        key="segmentation_intelligence_segment",
    )

    segmentation_table, segmentation_report = build_segmentation_table(
        df,
        selected_segment,
    )

    if segmentation_report.errors:
        st.error("Segmentation errors:")
        for error in segmentation_report.errors:
            st.write(error)

    else:
        sg1, sg2, sg3, sg4 = st.columns(4)

        sg1.metric(
            "Segments Analyzed",
            f"{segmentation_report.segments_analyzed:,}",
        )

        sg2.metric(
            "Total Expected Cost",
            f"${segmentation_report.total_expected_attrition_cost:,.0f}",
        )

        sg3.metric(
            "Largest Cost Segment",
            segmentation_report.largest_cost_segment,
        )

        sg4.metric(
            "Highest Risk Segment",
            segmentation_report.highest_risk_segment,
        )

        if segmentation_report.warnings:
            st.warning("Segmentation warnings:")
            for warning in segmentation_report.warnings:
                st.write(f"- {warning}")

        st.dataframe(
            segmentation_table,
            use_container_width=True,
        )

        if not segmentation_table.empty:

            top_cost_row = segmentation_table.sort_values(
                "total_expected_attrition_cost",
                ascending=False,
            ).iloc[0]

            top_risk_row = segmentation_table.sort_values(
                "avg_attrition_probability",
                ascending=False,
            ).iloc[0]

            st.subheader("Segmentation Interpretation")

            st.write(
                f"{top_cost_row['segment']} accounts for "
                f"{top_cost_row['share_of_total_cost_pct']:.1f}% of total expected "
                f"attrition cost, making it the largest economic exposure segment "
                f"under the selected segmentation view."
            )

            st.write(
                f"{top_risk_row['segment']} has the highest average predicted "
                f"attrition probability at {top_risk_row['avg_attrition_probability']:.1%}. "
                f"This does not imply an automated decision, but it indicates where "
                f"management may want to investigate retention drivers first."
            )

            st.write(
                "This segmentation view helps identify where workforce risk is concentrated "
                "inside the organization, beyond occupation-level analysis."
            )

        st.download_button(
            label="Download Workforce Segmentation Table",
            data=segmentation_table.to_csv(index=False).encode("utf-8"),
            file_name=f"hcrl_segmentation_by_{selected_segment}.csv",
            mime="text/csv",
        )

# =====================================================
# 13. ATTRITION DRIVER INTELLIGENCE
# =====================================================

st.header("13. Attrition Driver Intelligence")

driver_table, driver_report = build_attrition_driver_table(df)

if driver_report.errors:
    st.error("Attrition driver errors:")
    for error in driver_report.errors:
        st.write(error)
    driver_table = pd.DataFrame()

else:
    valid_driver_groups = {
        "Compensation",
        "Career Progression",
        "Employee Experience",
        "Manager Stability",
        "Workload",
        "Work Environment",
        "Travel / Commute Burden",
        "Training and Development",
        "Department",
        "Occupation",
        "Education",
        "Performance",
    }

    driver_table = driver_table[
        driver_table["driver_group"].isin(valid_driver_groups)
    ].copy()

    st.metric("Drivers Analyzed", f"{len(driver_table):,}")

    if driver_report.warnings:
        st.warning("Driver analysis warnings:")
        for warning in driver_report.warnings:
            st.write(f"- {warning}")

    st.subheader("Attrition Driver Evidence Table")
    st.dataframe(driver_table.head(25), use_container_width=True)

    st.subheader("Top Actionable Attrition Drivers")

    actionable_drivers = driver_table[
        driver_table["actionability"] == "Actionable"
    ].copy()

    top_drivers = actionable_drivers.head(10)[
        [
            "driver_rank",
            "driver_group",
            "driver_variable",
            "association_value",
            "direction",
            "actionability",
            "evidence_summary",
        ]
    ]

    st.dataframe(top_drivers, use_container_width=True)

    st.subheader("Driver Interpretation")

    for _, row in actionable_drivers.head(5).iterrows():

        direction_text = (
            "higher values are associated with higher predicted attrition risk"
            if row["association_value"] > 0
            else "higher values are associated with lower predicted attrition risk"
        )

        st.write(
            f"• **{row['driver_group']}** "
            f"({row['association_value']:.3f}): "
            f"{direction_text}."
        )

    st.info(
        "These business drivers are statistically associated with predicted attrition risk. "
        "The analysis does not establish causality and should not be interpreted as an "
        "automated employment recommendation."
    )

    st.download_button(
        label="Download Attrition Driver Table",
        data=driver_table.to_csv(index=False).encode("utf-8"),
        file_name="hcrl_attrition_driver_intelligence.csv",
        mime="text/csv",
    )

# =====================================================
# 14. INTERVENTION INTELLIGENCE
# =====================================================

st.header("14. Intervention Intelligence")

intervention_table, intervention_report = build_intervention_intelligence_table(
    driver_table=driver_table,
)

if intervention_report.errors:
    st.error("Intervention intelligence errors:")
    for error in intervention_report.errors:
        st.write(error)

else:
    st.metric(
        "Intervention Areas Identified",
        intervention_report.intervention_areas_identified,
    )

    if intervention_report.warnings:
        st.warning("Intervention intelligence warnings:")
        for warning in intervention_report.warnings:
            st.write(f"- {warning}")

    st.subheader("Evidence-Aligned Intervention Areas")

    st.dataframe(
        intervention_table,
        use_container_width=True,
    )

    st.download_button(
        label="Download Intervention Intelligence Table",
        data=intervention_table.to_csv(index=False).encode("utf-8"),
        file_name="hcrl_intervention_intelligence.csv",
        mime="text/csv",
    )


# =====================================================
# 15. INTERVENTION ECONOMICS
# =====================================================

st.header("15. Intervention Economics")

intervention_economics_table, intervention_economics_report = (
    build_intervention_economics_table(
        intervention_table=intervention_table,
        priority_table=priority_table,
    )
)

if intervention_economics_report.errors:
    st.error("Intervention economics errors:")
    for error in intervention_economics_report.errors:
        st.write(error)

else:
    st.metric(
        "Intervention Areas Analyzed",
        intervention_economics_report.intervention_areas_analyzed,
    )

    st.metric(
        "Total Modeled Exposure",
        f"${intervention_economics_report.total_modeled_exposure:,.0f}",
    )

    if intervention_economics_report.warnings:
        st.warning("Intervention economics warnings:")
        for warning in intervention_economics_report.warnings:
            st.write(f"- {warning}")

    st.subheader("Intervention-Linked Economic Exposure")

    st.dataframe(
        intervention_economics_table,
        use_container_width=True,
    )

    st.download_button(
        label="Download Intervention Economics Table",
        data=intervention_economics_table.to_csv(index=False).encode("utf-8"),
        file_name="hcrl_intervention_economics.csv",
        mime="text/csv",
    )


# =====================================================
# 16. WORKFORCE LEVERAGE INTELLIGENCE
# =====================================================

st.header("16. Workforce Leverage Intelligence")

leverage_table, leverage_report = build_workforce_leverage_table(
    driver_table
)

if leverage_report.errors:
    st.error("Leverage engine errors:")
    for error in leverage_report.errors:
        st.write(error)

else:
    st.metric(
        "Leverage Areas Identified",
        leverage_report.leverage_areas_identified,
    )

    if leverage_report.warnings:
        st.warning("Leverage warnings:")
        for warning in leverage_report.warnings:
            st.write(f"- {warning}")

    st.dataframe(leverage_table, use_container_width=True)

    st.download_button(
        label="Download Workforce Leverage Table",
        data=leverage_table.to_csv(index=False).encode("utf-8"),
        file_name="hcrl_workforce_leverage.csv",
        mime="text/csv",
    )

# =====================================================
# 17. WORKFORCE ACTION INTELLIGENCE
# =====================================================

st.header("17. Workforce Action Intelligence")

if (
    "intervention_economics_table" in locals()
    and intervention_economics_table is not None
    and not intervention_economics_table.empty
):

    action_df = intervention_economics_table.copy()

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

    missing_cols = [
        col for col in required_cols
        if col not in action_df.columns
    ]

    if missing_cols:
        st.error(f"Missing columns for Workforce Action Intelligence: {missing_cols}")

    else:
        action_df["actionability_sort"] = (
            action_df["actionability"]
            .map({"Actionable": 1, "Descriptive": 0})
            .fillna(0)
        )

        action_df = (
            action_df
            .sort_values(
                ["actionability_sort", "economic_attention_score"],
                ascending=[False, False],
            )
            .drop(columns=["actionability_sort"])
            .reset_index(drop=True)
        )

        action_df["action_rank"] = range(1, len(action_df) + 1)

        def evidence_strength_label(n_drivers):
            if n_drivers >= 4:
                return "Very High"
            if n_drivers >= 2:
                return "High"
            return "Moderate"

        def management_attention_label(rank, actionability):
            if actionability != "Actionable":
                return "Contextual Review"
            if rank <= 2:
                return "Immediate Management Review"
            if rank <= 5:
                return "Near-Term Management Review"
            return "Planned Review"

        management_questions = {
            "Career Progression": [
                "Are employees remaining too long without promotion?",
                "Are internal mobility paths visible and accessible?",
                "Are high-performing employees progressing internally?",
                "Are promotion patterns consistent across departments?",
            ],
            "Compensation": [
                "Is compensation competitive for high-risk roles?",
                "Are there signs of pay compression across job levels?",
                "Are salary increases aligned with retention-sensitive roles?",
                "Are long-term incentives being used effectively?",
            ],
            "Work Environment": [
                "Are low satisfaction scores concentrated in specific teams?",
                "Do high-risk groups report weaker job involvement?",
                "Are employee experience issues concentrated by manager or department?",
                "Are workload, recognition, and team climate being reviewed together?",
            ],
            "Manager Stability": [
                "Do high-risk groups share unstable manager relationships?",
                "Are manager changes concentrated in exposed workforce areas?",
                "Do teams with shorter manager tenure show higher modeled risk?",
                "Should manager continuity be reviewed for critical roles?",
            ],
            "Travel / Commute Burden": [
                "Are commute or travel expectations concentrated in high-risk groups?",
                "Can flexibility reduce avoidable workforce friction?",
                "Are location requirements aligned with role needs?",
                "Do travel-heavy roles show higher modeled attrition risk?",
            ],
            "Workload": [
                "Is overtime concentrated in high-risk roles or departments?",
                "Are staffing levels aligned with demand?",
                "Are workload spikes creating retention pressure?",
                "Should scheduling or capacity planning be reviewed?",
            ],
            "Training and Development": [
                "Are development opportunities reaching exposed workforce groups?",
                "Are training investments aligned with retention-sensitive roles?",
                "Do employees in high-risk areas receive enough skill development?",
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
                "Do educational profiles differ across high-risk groups?",
                "Is education acting as a descriptive workforce signal?",
                "Should education be reviewed only as context, not as a direct action lever?",
                "Are skill-development pathways more useful than education categories?",
            ],
        }

        suggested_actions = {
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
                "Supporting: Review whether issues are concentrated by manager or department.",
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
                "Supporting: Review whether high performers are receiving advancement opportunities.",
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

        business_risks = {
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

        action_df["evidence_strength"] = action_df["evidence_drivers"].apply(
            evidence_strength_label
        )

        action_df["management_attention"] = action_df.apply(
            lambda row: management_attention_label(
                row["action_rank"],
                row["actionability"],
            ),
            axis=1,
        )

        action_df["management_questions"] = action_df["driver_group"].apply(
            lambda x: " | ".join(management_questions.get(x, ["Review this workforce domain with HR leadership."]))
        )

        action_df["recommended_management_investigations"] = action_df["driver_group"].apply(
            lambda x: " | ".join(suggested_actions.get(x, ["Review this workforce domain before making decisions."]))
        )

        action_df["business_risks_if_ignored"] = action_df["driver_group"].apply(
            lambda x: " | ".join(business_risks.get(x, ["Potential workforce risk may remain unexplained."]))
        )

        st.metric("Action Areas Identified", len(action_df))

        st.warning(
            "Workforce Action Intelligence converts driver evidence and modeled exposure "
            "into management investigation priorities. It does not estimate ROI, causal savings, "
            "or prescribe mandatory actions."
        )

        top = action_df.iloc[0]

        st.subheader("Executive Summary")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Highest Priority", top["driver_group"])

        with c2:
            st.metric(
                "Linked Exposure",
                f"${top['exposure_linked_to_intervention_area']:,.0f}",
            )

        with c3:
            st.metric("Evidence Strength", top["evidence_strength"])

        with c4:
            st.metric("Management Attention", top["management_attention"])

        st.subheader("Executive Priority Dashboard")

        dashboard_cols = [
            "action_rank",
            "driver_group",
            "intervention_area",
            "actionability",
            "evidence_strength",
            "management_attention",
            "evidence_drivers",
            "supporting_variables",
            "exposure_linked_to_intervention_area",
            "economic_attention_score",
        ]

        st.dataframe(
            action_df[dashboard_cols],
            use_container_width=True,
        )

        st.subheader("Top Management Priority")

        st.success(
            f"""
            Highest-priority workforce domain: **{top['driver_group']}**

            Recommended investigation area: **{top['intervention_area']}**

            Management attention: **{top['management_attention']}**

            Evidence strength: **{top['evidence_strength']}**

            Modeled exposure linked to this area: **${top['exposure_linked_to_intervention_area']:,.0f}**

            Supporting variables: **{top['supporting_variables']}**
            """
        )

        st.subheader("Priority Briefings")

        for _, row in action_df.head(5).iterrows():
            questions = management_questions.get(row["driver_group"], [])
            actions = suggested_actions.get(row["driver_group"], [])
            risks = business_risks.get(row["driver_group"], [])

            st.markdown(
                f"""
## Priority #{int(row['action_rank'])} — {row['driver_group']}

### Why this appeared

{row['driver_group']} appears as a management priority because it is supported by **{int(row['evidence_drivers'])} evidence driver(s)** and is linked to approximately **${row['exposure_linked_to_intervention_area']:,.0f}** of modeled workforce exposure.

**Evidence strength:** {row['evidence_strength']}

**Management attention:** {row['management_attention']}

**Supporting variables:** {row['supporting_variables']}

### What management should investigate
"""
            )

            for q in questions:
                st.write(f"- {q}")

            st.markdown("### Recommended management investigations")

            for a in actions:
                st.write(f"- {a}")

            st.markdown("### Potential business risks if ignored")

            for r in risks:
                st.write(f"- {r}")

            st.info(
                "This is an evidence-aligned management briefing. "
                "It is not a causal recommendation, ROI estimate, or guaranteed savings forecast."
            )

        st.download_button(
            label="Download Workforce Action Intelligence",
            data=action_df.to_csv(index=False).encode("utf-8"),
            file_name="hcrl_workforce_action_intelligence.csv",
            mime="text/csv",
        )

else:
    st.info("Intervention Economics must run before Workforce Action Intelligence.")

with st.expander("Methodology and Limitations"):
    st.write(
        """
HCRL is a quantitative workforce transformation intelligence platform.

The system separates workforce data validation, occupation mapping, task
intelligence, attrition risk estimation, economic exposure modeling, decision
intelligence, and intervention economics into independent modules.

The platform does not make firing recommendations.

Current enterprise limitations:
- Company-specific attrition prediction requires historical separation outcomes.
- If no historical attrition outcomes exist, an external labor-market baseline
  risk model is required.
- Expected cost estimation requires replacement-cost inputs or an externally
  calibrated replacement-cost model.
- Intervention ROI requires company-supplied or externally validated cost and
  benefit assumptions.
- O*NET mapping should be reviewed when match status is review_required.
- Task intelligence depends on successful O*NET occupation mapping.
        """
    )

st.markdown("---")
st.caption("Human Capital Risk Lab | Enterprise Workforce Transformation Intelligence")
