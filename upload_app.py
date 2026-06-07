import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

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
    st.info(
        "IBM HR Attrition dataset detected. "
        "Risk scores are generated using a trained model fitted on the uploaded dataset."
    )

    df["attrition_target"] = df["Attrition"].map({
        "Yes": 1,
        "No": 0
    })

    model_features = [
        "Age", "BusinessTravel", "Department", "DistanceFromHome",
        "Education", "EnvironmentSatisfaction", "Gender", "JobInvolvement",
        "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
        "MonthlyIncome", "NumCompaniesWorked", "OverTime",
        "PercentSalaryHike", "PerformanceRating",
        "RelationshipSatisfaction", "StockOptionLevel",
        "TotalWorkingYears", "TrainingTimesLastYear",
        "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager"
    ]

    missing_model_features = [
        c for c in model_features if c not in df.columns
    ]

    if missing_model_features:
        st.error(
            f"Dataset missing required model features: {missing_model_features}"
        )
        st.stop()

    X_model = df[model_features].copy()
    y_model = df["attrition_target"].copy()

    categorical_features = X_model.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X_model.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    attrition_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ])

    attrition_model.fit(X_model, y_model)

    df["predicted_risk"] = attrition_model.predict_proba(X_model)[:, 1]

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
# AI Workforce Strategy Reference
# =====================

@st.cache_data
def load_ai_strategy_table():
    return pd.read_csv("hcrl_ai_workforce_strategy_table.csv")

ai_ref = load_ai_strategy_table()

def clean_text(x):
    return str(x).lower().strip()

ai_ref["match_title"] = ai_ref["Title"].apply(clean_text)

ibm_to_onet_role_map = {
    "Sales Executive": "Sales Managers",
    "Sales Representative": "Sales Representatives, Wholesale and Manufacturing, Except Technical and Scientific Products",
    "Research Scientist": "Computer and Information Research Scientists",
    "Laboratory Technician": "Medical and Clinical Laboratory Technicians",
    "Manufacturing Director": "Industrial Production Managers",
    "Healthcare Representative": None,
    "Manager": "General and Operations Managers",
    "Research Director": "Natural Sciences Managers",
    "Human Resources": "Human Resources Managers"
}
role_col = None
for possible_col in ["JobRole", "job_role", "Title", "occupation", "Occupation"]:
    if possible_col in df.columns:
        role_col = possible_col
        break

def match_ai_strategy(role):
    role_clean = clean_text(role)
    mapped_title = ibm_to_onet_role_map.get(str(role))

    if mapped_title is not None:
        mapped_clean = clean_text(mapped_title)

        mapped_exact = ai_ref[ai_ref["match_title"] == mapped_clean]
        if len(mapped_exact) > 0:
            return mapped_exact.iloc[0]
            
    exact = ai_ref[ai_ref["match_title"] == role_clean]
    if len(exact) > 0:
        return exact.iloc[0]

    contains = ai_ref[ai_ref["match_title"].str.contains(role_clean, na=False)]
    if len(contains) > 0:
        return contains.iloc[0]

    reverse_contains = ai_ref[ai_ref["match_title"].apply(lambda x: x in role_clean)]
    if len(reverse_contains) > 0:
        return reverse_contains.iloc[0]

    return pd.Series({
        "Title": np.nan,
        "ai_exposure_score": np.nan,
        "workforce_strategy_category": "Unmatched",
        "digital_intensity": np.nan,
        "cognitive_complexity": np.nan,
        "human_interaction": np.nan,
        "physical_work": np.nan
    })

if role_col is not None:
    ai_matches = df[role_col].apply(match_ai_strategy)

    df["matched_onet_title"] = ai_matches["Title"].values
    df["ai_exposure_score"] = ai_matches["ai_exposure_score"].values
    df["workforce_strategy_category"] = ai_matches["workforce_strategy_category"].values
    df["digital_intensity"] = ai_matches["digital_intensity"].values
    df["cognitive_complexity"] = ai_matches["cognitive_complexity"].values
    df["human_interaction"] = ai_matches["human_interaction"].values
    df["physical_work"] = ai_matches["physical_work"].values
else:
    df["workforce_strategy_category"] = "No job role column detected"


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
# Risk concentration
# =====================

st.header("Workforce Risk Concentration")

if "JobRole" in df.columns:
    concentration_col = "JobRole"
elif "job_role" in df.columns:
    concentration_col = "job_role"
else:
    concentration_col = segment_col

risk_concentration = (
    df.groupby(concentration_col)
    .agg(
        avg_risk=("predicted_risk", "mean"),
        n_workers=(concentration_col, "count"),
        total_expected_exposure=("baseline_expected_cost", "sum")
    )
)

risk_concentration["risk_share_pct"] = (
    risk_concentration["total_expected_exposure"]
    / risk_concentration["total_expected_exposure"].sum()
    * 100
)

risk_concentration = risk_concentration.sort_values(
    "risk_share_pct",
    ascending=False
)

display_concentration = risk_concentration.copy()
display_concentration["avg_risk"] = display_concentration["avg_risk"].map(lambda x: f"{x:.1%}")
display_concentration["total_expected_exposure"] = display_concentration["total_expected_exposure"].map(lambda x: f"${x:,.0f}")
display_concentration["risk_share_pct"] = display_concentration["risk_share_pct"].map(lambda x: f"{x:.2f}%")

st.dataframe(display_concentration, use_container_width=True)

top_risk_concentration = risk_concentration.head(10)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.barh(
    top_risk_concentration.index.astype(str),
    top_risk_concentration["risk_share_pct"]
)
ax2.set_xlabel("Share of Total Expected Workforce Risk (%)")
ax2.set_ylabel(concentration_col)
ax2.set_title("Top Workforce Segments by Share of Total Risk")
ax2.invert_yaxis()
st.pyplot(fig2)

st.info(
    "This view shows which workforce segments contribute the largest share of total expected workforce risk. "
    "Unlike equal-sized risk buckets, this identifies where risk is actually concentrated."
)

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
# AI Workforce Transformation
# =====================

st.header("AI Workforce Transformation")

if role_col is None:
    st.warning("No job-role column detected. Add a column such as JobRole, job_role, Title, occupation, or Occupation.")
else:
    matched_rate = df["matched_onet_title"].notna().mean()

a1, a2, a3 = st.columns(3)

a1.metric(
    "AI Mapping Coverage",
    f"{matched_rate:.1%}"
)

a2.metric(
    "Average AI Exposure",
    f"{mapped_df['ai_exposure_score'].mean():.2f}"
)

a3.metric(
    "Most Common AI Strategy",
    mapped_df["workforce_strategy_category"].mode()[0]
)


st.write(
    "This section maps workforce roles to O*NET-based occupational AI exposure categories. "
    "The categories represent workforce transformation pathways, not firing recommendations."
)
    
# Only occupations successfully matched to O*NET

mapped_df = df[
    df["workforce_strategy_category"] != "Unmatched"
].copy()

    strategy_summary = (
        mapped_df.groupby("workforce_strategy_category")
        .agg(
            n_workers=("workforce_strategy_category", "count"),
            avg_ai_exposure=("ai_exposure_score", "mean"),
            avg_attrition_risk=("predicted_risk", "mean"),
            avg_stressed_cost=("stressed_expected_cost", "mean")
        )
        .sort_values("n_workers", ascending=False)
    )

    display_strategy = strategy_summary.copy()
    display_strategy["avg_ai_exposure"] = display_strategy["avg_ai_exposure"].map(lambda x: f"{x:.2f}")
    display_strategy["avg_attrition_risk"] = display_strategy["avg_attrition_risk"].map(lambda x: f"{x:.1%}")
    display_strategy["avg_stressed_cost"] = display_strategy["avg_stressed_cost"].map(lambda x: f"${x:,.0f}")

    st.subheader("Workforce Strategy Summary")
    st.dataframe(display_strategy, use_container_width=True)

    st.subheader("Role-Level AI Strategy Table")

    role_ai_summary = (
        mapped_df.groupby(role_col)
        .agg(
            matched_onet_title=("matched_onet_title", "first"),
            workforce_strategy_category=("workforce_strategy_category", "first"),
            avg_ai_exposure=("ai_exposure_score", "mean"),
            avg_attrition_risk=("predicted_risk", "mean"),
            avg_stressed_cost=("stressed_expected_cost", "mean"),
            n_workers=(role_col, "count")
        )
        .sort_values("avg_ai_exposure", ascending=False)
    )

    display_role_ai = role_ai_summary.copy()
    display_role_ai["avg_ai_exposure"] = display_role_ai["avg_ai_exposure"].map(lambda x: f"{x:.2f}")
    display_role_ai["avg_attrition_risk"] = display_role_ai["avg_attrition_risk"].map(lambda x: f"{x:.1%}")
    display_role_ai["avg_stressed_cost"] = display_role_ai["avg_stressed_cost"].map(lambda x: f"${x:,.0f}")

    st.dataframe(display_role_ai, use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    strategy_summary["n_workers"].plot(kind="bar", ax=ax3)
    ax3.set_title("Workforce Distribution by AI Strategy Category")
    ax3.set_xlabel("Strategy Category")
    ax3.set_ylabel("Number of Workers")
    plt.xticks(rotation=35, ha="right")
    st.pyplot(fig3)

    st.info(
        "Interpretation: AI-Augmentable roles may benefit from productivity tools; "
        "Routine Automation Candidates may be suitable for workflow redesign; "
        "Human-Centered and Physical/Field Protected roles require more careful human-centered transition planning."
    )

# =====================
# HCRL Decision Engine
# =====================

st.divider()
st.header("HCRL Decision Engine")

@st.cache_data
def load_decision_report():
    return pd.read_csv("hcrl_v1_integrated_decision_report.csv")

try:
    decision_report = load_decision_report()

    st.write(
        "This section combines workforce attrition risk, economic exposure, root-cause drivers, "
        "and O*NET-based AI strategy categories into an executive decision report."
    )

    st.subheader("Integrated Workforce Decision Report")

    display_decision = decision_report.copy()

    if "risk_share_pct" in display_decision.columns:
        display_decision["risk_share_pct"] = display_decision["risk_share_pct"].map(
            lambda x: f"{x:.2f}%"
        )

    if "ai_exposure_score" in display_decision.columns:
        display_decision["ai_exposure_score"] = display_decision["ai_exposure_score"].map(
            lambda x: "AI mapping unavailable" if pd.isna(x) else f"{x:.2f}"
        )

    st.dataframe(display_decision, use_container_width=True)

    st.subheader("Top Strategic Priorities")

    top_priorities = decision_report.sort_values("risk_rank").head(5)

    for _, row in top_priorities.iterrows():
        st.markdown(f"""
        **{row['JobRole']}**

        - Risk Priority Rank: #{int(row['risk_rank'])}
        - Share of Total Workforce Risk: {row['risk_share_pct']:.2f}%
        - Primary Risk Driver: {row['Driver']}
        - AI Strategy Category: {row['workforce_strategy_category']}
        - Recommendation: {row['final_hcrl_recommendation']}
        """)

    st.download_button(
        label="Download HCRL Integrated Decision Report",
        data=decision_report.to_csv(index=False).encode("utf-8"),
        file_name="hcrl_v1_integrated_decision_report.csv",
        mime="text/csv"
    )

except FileNotFoundError:
    st.warning(
        "HCRL integrated decision report not found. "
        "Upload `hcrl_v1_integrated_decision_report.csv` to the GitHub repository."
    )

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
HCRL analyzes workforce risk by combining employee turnover patterns, workforce economics, and occupational AI exposure data. The platform identifies which workforce segments contribute most to organizational risk, estimates potential business impact, and recommends targeted interventions to improve workforce stability, productivity, and long-term resilience. AI-related recommendations focus on workforce transformation, augmentation, and reskilling opportunities rather than employee replacement.
        """
    )

st.markdown("---")
st.caption(
    "Human Capital Risk Lab (HCRL) | Upload-Based Workforce Risk Analytics Prototype"
)
