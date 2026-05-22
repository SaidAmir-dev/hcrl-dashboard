import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="HCRL Dashboard", layout="wide")

st.title("Human Capital Risk Lab")
st.subheader("Quantitative Workforce Risk Analytics")

st.write(
    "This dashboard estimates workforce attrition risk and expected cost exposure "
    "under baseline and stressed labor-market scenarios. The current MVP uses public CPS data "
    "and is intended for analytical demonstration, not HR decision-making."
)

df = pd.read_csv("hcrl_model_dataset_v1.csv")

industry_labels = {
    4180: "Hospitals",
    2990: "Manufacturing",
    6672: "Construction",
    4680: "Education",
    8562: "Restaurants",
    5670: "Retail Trade",
    8563: "Food Services",
    7570: "Transportation",
    4195: "Healthcare",
    5190: "Finance"
}

df["industry_name"] = df["industry"].map(industry_labels).fillna(df["industry"].astype(str))

st.sidebar.header("Scenario Controls")

stress = st.sidebar.slider("Stress Multiplier", 1.0, 2.0, 1.2, 0.05)
lambda_multiplier = st.sidebar.slider("Replacement Cost Multiplier", 0.1, 1.5, 0.5, 0.1)

df["stressed_risk"] = np.minimum(1, df["predicted_risk"] * stress)
df["replacement_cost"] = lambda_multiplier * df["annual_wage_proxy"]
df["stressed_expected_cost"] = df["stressed_risk"] * df["replacement_cost"]

col1, col2, col3 = st.columns(3)

col1.metric("Average Predicted Risk", f"{df['predicted_risk'].mean():.1%}")
col2.metric("Average Stressed Risk", f"{df['stressed_risk'].mean():.1%}")
col3.metric("Average Stressed Expected Cost", f"${df['stressed_expected_cost'].mean():,.0f}")

st.divider()

st.header("Executive Summary")

baseline_cost = df["replacement_cost"].mul(df["predicted_risk"]).mean()
stressed_cost = df["stressed_expected_cost"].mean()
increase = stressed_cost - baseline_cost

st.write(
    f"""
Under the selected stress scenario, average expected attrition exposure rises from
**${baseline_cost:,.0f}** to **${stressed_cost:,.0f}**,
an increase of **${increase:,.0f}** per worker observation.
"""
)

st.divider()

industry_summary = (
    df.groupby("industry_name")
    .agg(
        avg_risk=("predicted_risk", "mean"),
        avg_stressed_risk=("stressed_risk", "mean"),
        avg_stressed_cost=("stressed_expected_cost", "mean"),
        n_workers=("industry_name", "count")
    )
    .sort_values("avg_stressed_cost", ascending=False)
    .head(10)
)

industry_display = industry_summary.copy()
industry_display["avg_risk"] = industry_display["avg_risk"].map(lambda x: f"{x:.1%}")
industry_display["avg_stressed_risk"] = industry_display["avg_stressed_risk"].map(lambda x: f"{x:.1%}")
industry_display["avg_stressed_cost"] = industry_display["avg_stressed_cost"].map(lambda x: f"${x:,.0f}")

st.header("Top Industries by Workforce Risk Exposure")
st.dataframe(industry_display, use_container_width=True)

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(industry_summary.index.astype(str), industry_summary["avg_stressed_cost"])
ax.set_xlabel("Industry")
ax.set_ylabel("Average Stressed Expected Cost")
ax.set_title("Top Industries by Workforce Risk Exposure")
plt.xticks(rotation=35, ha="right")
st.pyplot(fig)

st.caption(
    "MVP note: Industry labels are manually mapped for selected top industry codes. "
    "Future versions should use the official CPS/IPUMS industry label dictionary."
)
