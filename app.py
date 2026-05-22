import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="HCRL Dashboard",
    layout="wide"
)

st.title("Human Capital Risk Lab")
st.subheader(
    "Quantitative Workforce Risk Analytics"
)

# Load dataset
df = pd.read_csv(
    "hcrl_model_dataset_v1.csv"
)

# Sidebar controls
st.sidebar.header("Scenario Controls")

stress = st.sidebar.slider(
    "Stress Multiplier",
    1.0,
    2.0,
    1.2,
    0.05
)

lambda_multiplier = st.sidebar.slider(
    "Replacement Cost Multiplier",
    0.1,
    1.5,
    0.5,
    0.1
)

# Stress calculations
df["stressed_risk"] = np.minimum(
    1,
    df["predicted_risk"] * stress
)

df["replacement_cost"] = (
    lambda_multiplier *
    df["annual_wage_proxy"]
)

df["stressed_expected_cost"] = (
    df["stressed_risk"] *
    df["replacement_cost"]
)

# KPIs
col1, col2 = st.columns(2)

col1.metric(
    "Average Predicted Risk",
    round(df["predicted_risk"].mean(), 3)
)

col2.metric(
    "Average Stressed Expected Cost",
    round(df["stressed_expected_cost"].mean(), 2)
)

# Industry aggregation
industry_summary = (
    df.groupby("industry")
    .agg(
        avg_risk=("predicted_risk", "mean"),
        avg_stressed_cost=("stressed_expected_cost", "mean"),
        n_workers=("industry", "count")
    )
    .sort_values(
        "avg_stressed_cost",
        ascending=False
    )
    .head(10)
)

st.write(
    "Top Industries by Workforce Risk Exposure"
)

st.dataframe(industry_summary)

# Chart
fig, ax = plt.subplots(figsize=(10,5))

ax.bar(
    industry_summary.index.astype(str),
    industry_summary["avg_stressed_cost"]
)

ax.set_xlabel("Industry Code")
ax.set_ylabel("Avg Stressed Expected Cost")

ax.set_title(
    "Top Industries by Workforce Risk Exposure"
)

st.pyplot(fig)