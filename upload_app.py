import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="HCRL Upload Prototype", layout="wide")

st.title("Human Capital Risk Lab")
st.subheader("Upload-Based Workforce Risk Analytics Prototype")

st.write(
    "Upload a workforce dataset to estimate attrition risk, expected turnover exposure, "
    "and stressed workforce cost under different labor-market scenarios."
)

uploaded_file = st.file_uploader(
    "Upload Workforce Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom workforce dataset uploaded successfully.")
else:
    df = pd.read_csv("hcrl_model_dataset_v1.csv")
    st.info("No custom file uploaded. Using default HCRL demo dataset.")

# =====================
# Auto-detect IBM HR dataset
# =====================

if "Attrition" in df.columns and "MonthlyIncome" in df.columns:
    st.info("IBM HR Attrition dataset detected. Variables were automatically mapped.")

    df["predicted_risk"] = np.where(
        df["Attrition"].astype(str).str.lower() == "yes",
        0.75,
        0.25
    )

    df["annual_wage_proxy"] = df["MonthlyIncome"] * 12

    if "Department" in df.columns:
        df["department"] = df["Department"]

    if "JobRole" in df.columns:
        df["job_role"] = df["JobRole"]

    if "EmployeeNumber" in df.columns:
        df["employee_id"] = df["EmployeeNumber"]

# =====================
# Required columns
# =====================

required_cols = ["predicted_risk", "annual_wage_proxy"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.write(
        "Your dataset must include either `predicted_risk` and `annual_wage_proxy`, "
        "or IBM-style columns `Attrition` and `MonthlyIncome`."
    )
    st.stop()

# =====================
# Optional segment columns
# =====================

if "industry_name" not in df.columns:
    if "industry" in df.columns:
        df["industry_name"] = df["industry"].astype(str)
    elif "Department" in df.columns:
        df["industry_name"] = df["Department"]
    else:
        df["industry_name"] = "Unknown"

if "department" not in df.columns:
    if "Department" in df.columns:
        df["department"] = df["Department"]
    else:
        df["department"] = df["industry_name"]

# =====================
# Sidebar
# =====================

st.sidebar.header("Scenario Controls")

company_name = st.sidebar.text_input("Company Name", "Demo Organization")

stress = st.sidebar.slider(
    "Stress Multiplier",
    min_value=1.0,
    max_value=2.0,
    value=1.2,
    step=0.05
)

lambda_multiplier = st.sidebar.slider(
    "Replacement Cost Multiplier",
    min_value=0.1,
    max_value=1.5,
    value=0.5,
    step=0.1
)

# =====================
# Calculations
# =====================

df["predicted_risk"] = pd.to_numeric(df["predicted_risk"], errors="coerce")
df["annual_wage_proxy"] = pd.to_numeric(df["annual_wage_proxy"], errors="coerce")

df = df.dropna(subset=["predicted_risk", "annual_wage_proxy"]).copy()
df["predicted_risk"] = df["predicted_risk"].clip(0, 1)

df["stressed_risk"] = np.minimum(1, df["predicted_risk"] * stress)
df["replacement_cost"] = lambda_multiplier * df["annual_wage_proxy"]
df["baseline_expected_cost"] = df["predicted_risk"] * df["replacement_cost"]
df["stressed_expected_cost"] = df["stressed_risk"] * df["replacement_cost"]

# =====================
# Header
# =====================

st.header(f"{company_name} Workforce Risk Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Employees Analyzed", f"{len(df):,}")
col2.metric("Average Risk", f"{df['predicted_risk'].mean():.1%}")
col3.metric("Average Stressed Risk", f"{df['stressed_risk'].mean():.1%}")
col4.metric("Avg Stressed Exposure", f"${df['stressed_expected_cost'].mean():,.0f}")

st.divider()

# =====================
# Executive Summary
# =====================

baseline_cost = df["baseline_expected_cost"].mean()
stressed_cost = df["stressed_expected_cost"].mean()
increase = stressed_cost - baseline_cost

st.header("Executive Summary")

st.write(
    f"For {company_name}, average expected attrition exposure rises from "
    f"\${baseline_cost:,.0f} to \${stressed_cost:,.0f} under the selected stress scenario. "
    f"This represents an increase of \${increase:,.0f} per employee observation."
)

st.info(
    f"Average workforce risk increases from {df['predicted_risk'].mean():.1%} "
    f"to {df['stressed_risk'].mean():.1%} under the selected scenario."
)

st.divider()

# =====================
# Segment selection
# =====================

st.header("Workforce Segment Risk Exposure")

possible_segments = [
    col for col in [
        "department",
        "industry_name",
        "Department",
        "JobRole",
        "job_role",
        "EducationField",
        "BusinessTravel",
        "MaritalStatus"
    ]
    if col in df.columns
]

segment_col = st.selectbox(
    "Choose segmentation variable",
    possible_segments
)

segment_summary = (
    df.groupby(segment_col)
    .agg(
        avg_risk=("predicted_risk", "mean"),
        avg_stressed_risk=("stressed_risk", "mean"),
        avg_expected_cost=("baseline_expected_cost", "mean"),
        avg_stressed_cost=("stressed_expected_cost", "mean"),
        n_workers=(segment_col, "count")
    )
    .sort_values("avg_stressed_cost", ascending=False)
)

display_summary = segment_summary.copy()
display_summary["avg_risk"] = display_summary["avg_risk"].map(lambda x: f"{x:.1%}")
display_summary["avg_stressed_risk"] = display_summary["avg_stressed_risk"].map(lambda x: f"{x:.1%}")
display_summary["avg_expected_cost"] = display_summary["avg_expected_cost"].map(lambda x: f"${x:,.0f}")
display_summary["avg_stressed_cost"] = display_summary["avg_stressed_cost"].map(lambda x: f"${x:,.0f}")

st.dataframe(display_summary, use_container_width=True)

top_segments = segment_summary.head(10)

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(top_segments.index.astype(str), top_segments["avg_stressed_cost"])
ax.set_xlabel(segment_col)
ax.set_ylabel("Average Stressed Expected Cost")
ax.set_title("Top Workforce Segments by Stressed Attrition Exposure")
plt.xticks(rotation=35, ha="right")
st.pyplot(fig)

st.divider()

# =====================
# Risk segmentation
# =====================

st.header("Risk Segmentation")

df["risk_bucket"] = pd.cut(
    df["predicted_risk"],
    bins=[0, 0.3, 0.6, 1],
    labels=["Low Risk", "Medium Risk", "High Risk"],
    include_lowest=True
)

bucket_counts = df["risk_bucket"].value_counts().sort_index()

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.pie(bucket_counts, labels=bucket_counts.index, autopct="%1.1f%%")
ax2.set_title("Workforce Risk Distribution")
st.pyplot(fig2)

st.divider()

# =====================
# Segment drill-down
# =====================

st.header("Segment Drill-Down")

selected_segment = st.selectbox(
    "Select segment",
    segment_summary.index.astype(str).tolist()
)

segment_data = df[df[segment_col].astype(str) == selected_segment]

d1, d2, d3, d4 = st.columns(4)

d1.metric("Workers", f"{len(segment_data):,}")
d2.metric("Avg Risk", f"{segment_data['predicted_risk'].mean():.1%}")
d3.metric("Avg Stressed Risk", f"{segment_data['stressed_risk'].mean():.1%}")
d4.metric("Avg Stressed Cost", f"${segment_data['stressed_expected_cost'].mean():,.0f}")

st.write(
    f"The selected segment has an average baseline attrition risk of "
    f"{segment_data['predicted_risk'].mean():.1%}. Under the selected stress scenario, "
    f"average expected attrition exposure is "
    f"${segment_data['stressed_expected_cost'].mean():,.0f} per employee observation."
)

st.divider()

# =====================
# Download report
# =====================

st.header("Download Report")

st.download_button(
    label="Download Segment Risk Report",
    data=segment_summary.to_csv().encode("utf-8"),
    file_name="hcrl_segment_risk_report.csv",
    mime="text/csv"
)

st.download_button(
    label="Download Full Analyzed Dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="hcrl_analyzed_workforce_data.csv",
    mime="text/csv"
)

# =====================
# Methodology
# =====================

with st.expander("Methodology"):
    st.write(
        """
        This upload-based prototype estimates workforce attrition exposure using three core inputs:
        predicted attrition risk, annual wage, and a replacement-cost multiplier.

        For the IBM HR Attrition dataset, the prototype maps:
        - MonthlyIncome to annual wage by multiplying by 12
        - Attrition to a simplified risk proxy

        Expected Attrition Cost = Predicted Risk × Replacement Cost

        Replacement Cost = Annual Wage × Replacement Cost Multiplier

        The stress scenario increases predicted attrition probabilities by the selected multiplier,
        capped at 100%.

        This MVP is intended for analytical demonstration and does not make individual HR,
        hiring, firing, or performance decisions.
        """
    )

st.markdown("---")
st.caption(
    "Human Capital Risk Lab (HCRL) | Upload-Based Workforce Risk Analytics Prototype"
)
