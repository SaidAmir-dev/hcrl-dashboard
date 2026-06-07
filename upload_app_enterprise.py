import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from hcrl_schema import standardize_workforce_data
from hcrl_risk_engine import estimate_attrition_risk
from hcrl_cost_engine import estimate_expected_cost
from hcrl_onet_mapper import map_to_onet
from hcrl_decision_engine import build_segment_exposure, build_decision_support_table

st.set_page_config(page_title="HCRL Enterprise Architecture", layout="wide")

st.title("Human Capital Risk Lab")
st.subheader("Enterprise Workforce Risk Architecture")
st.write(
    "This version separates schema validation, risk modeling, cost exposure, O*NET mapping, "
    "and decision-support logic. It avoids IBM-specific assumptions as the core architecture."
)

uploaded_file = st.file_uploader("Upload Workforce Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Upload a company workforce CSV to begin. The app will not fabricate demo analytics when no data is supplied.")
    st.stop()

raw_df = pd.read_csv(uploaded_file)
df, schema_report = standardize_workforce_data(raw_df)

st.header("1. Data Validation Layer")
st.write(f"Detected source type: `{schema_report.source_type}`")
st.json(schema_report.mapped_columns)

for warning in schema_report.warnings:
    st.warning(warning)
for error in schema_report.errors:
    st.error(error)

if not schema_report.is_valid_for_risk:
    st.stop()

st.header("2. Workforce Risk Engine")
try:
    df, risk_report = estimate_attrition_risk(
        df=df,
        feature_columns=schema_report.model_feature_columns,
    )
    st.write(f"Risk method: `{risk_report.method}`")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Observations", f"{risk_report.n_observations:,}")
    if risk_report.event_rate is not None:
        metric_cols[1].metric("Observed Event Rate", f"{risk_report.event_rate:.1%}")
    if risk_report.auc_oof is not None:
        metric_cols[2].metric("OOF ROC-AUC", f"{risk_report.auc_oof:.3f}")
    if risk_report.brier_oof is not None:
        metric_cols[3].metric("OOF Brier Score", f"{risk_report.brier_oof:.3f}")
    for warning in risk_report.warnings:
        st.warning(warning)
except Exception as e:
    st.error(f"Risk engine failed: {e}")
    st.stop()

st.header("3. Economic Exposure Engine")
try:
    df, cost_report = estimate_expected_cost(df)
    st.write(f"Cost method: `{cost_report.method}`")
    for warning in cost_report.warnings:
        st.warning(warning)
except Exception as e:
    st.error(f"Cost engine failed: {e}")

st.header("4. O*NET Occupation Intelligence Layer")
onet_file = st.file_uploader("Optional: Upload O*NET occupation feature table", type=["csv"], key="onet")
if onet_file is not None:
    onet_ref = pd.read_csv(onet_file)
    try:
        df, onet_report = map_to_onet(df, onet_ref)
        st.metric("O*NET Mapping Coverage", f"{onet_report.coverage:.1%}")
        st.write(onet_report.note)
        st.write(
            f"Exact matches: {onet_report.exact_matches:,}; fuzzy review-required matches: {onet_report.fuzzy_matches:,}; unmatched: {onet_report.unmatched:,}."
        )
    except Exception as e:
        st.error(f"O*NET mapping failed: {e}")
else:
    st.info("O*NET mapping skipped. Upload an O*NET reference file to activate occupation intelligence.")

st.header("5. Workforce Segment Exposure")
possible_segments = [
    c for c in ["department", "job_title", "occupation_code", "location"]
    if c in df.columns
]
if not possible_segments:
    st.warning("No segment columns detected.")
    st.stop()

segment_col = st.selectbox("Choose segmentation variable", possible_segments)
segment_summary = build_segment_exposure(df, segment_col)
st.dataframe(segment_summary, use_container_width=True)

if "total_expected_attrition_cost" in segment_summary.columns:
    plot_df = segment_summary.head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(plot_df[segment_col].astype(str), plot_df["total_expected_attrition_cost"])
    ax.set_xlabel("Total Expected Attrition Cost")
    ax.set_ylabel(segment_col)
    ax.set_title("Top Segments by Expected Attrition Cost")
    ax.invert_yaxis()
    st.pyplot(fig)

st.header("6. Decision-Support Table")
decision_table = build_decision_support_table(df, segment_col)
st.dataframe(decision_table, use_container_width=True)

st.header("7. Download Audit Outputs")
st.download_button(
    "Download Analyzed Workforce Data",
    df.to_csv(index=False).encode("utf-8"),
    file_name="hcrl_analyzed_workforce_data.csv",
    mime="text/csv",
)
st.download_button(
    "Download Segment Decision-Support Table",
    decision_table.to_csv(index=False).encode("utf-8"),
    file_name="hcrl_segment_decision_support.csv",
    mime="text/csv",
)

with st.expander("Methodology Notes"):
    st.write(
        "HCRL estimates attrition as a probabilistic separation event where outcome data are available. "
        "Expected cost is only estimated when replacement-cost inputs are supplied directly by the company or derived from company-supplied replacement-cost multipliers. "
        "O*NET mapping is occupational context and should be confirmed with SOC codes for enterprise deployment."
    )
