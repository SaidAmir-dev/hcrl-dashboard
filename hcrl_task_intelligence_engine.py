"""HCRL task intelligence engine.

This module connects mapped O*NET occupations to their task portfolio.

It uses two HCRL-generated files:

1. data/hcrl_occupation_task_portfolio.csv
   Task-level table:
   occupation -> task -> importance/relevance/frequency/DWA links

2. data/hcrl_occupation_task_summary.csv
   Occupation-level task summary:
   occupation -> task counts and aggregate task statistics

Purpose:
This is the foundation for:
- AI exposure
- strategic human capital importance
- augmentation potential
- automation feasibility
- workforce redesign analysis

Important:
This module does not invent task scores.
It only joins company occupations to O*NET task evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class TaskIntelligenceReport:
    task_summary_available: bool
    task_portfolio_available: bool
    matched_occupations: int
    unmatched_occupations: int
    warnings: List[str]
    errors: List[str]


def _find_code_column(df: pd.DataFrame) -> Optional[str]:
    for col in [
        "matched_onet_code",
        "O*NET-SOC Code",
        "onet_soc_code",
        "occupation_code",
        "soc_code",
    ]:
        if col in df.columns:
            return col
    return None


def _standardize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def load_task_summary(
    path: str = "data/hcrl_occupation_task_summary.csv",
) -> pd.DataFrame:
    summary = pd.read_csv(path)

    if "O*NET-SOC Code" not in summary.columns:
        raise ValueError("Task summary must contain 'O*NET-SOC Code'.")

    summary["O*NET-SOC Code"] = _standardize_code(summary["O*NET-SOC Code"])

    return summary


def load_task_portfolio(
    path: str = "data/hcrl_occupation_task_portfolio.csv",
) -> pd.DataFrame:
    portfolio = pd.read_csv(path)

    required = ["O*NET-SOC Code", "Task ID", "Task"]

    missing = [col for col in required if col not in portfolio.columns]

    if missing:
        raise ValueError(
            f"Task portfolio missing required columns: {missing}"
        )

    portfolio["O*NET-SOC Code"] = _standardize_code(portfolio["O*NET-SOC Code"])

    return portfolio


def attach_task_summary(
    workforce_df: pd.DataFrame,
    task_summary: pd.DataFrame,
) -> Tuple[pd.DataFrame, TaskIntelligenceReport]:

    out = workforce_df.copy()
    warnings: List[str] = []
    errors: List[str] = []

    code_col = _find_code_column(out)

    if code_col is None:
        errors.append(
            "No O*NET/SOC code column found. Task intelligence cannot be attached."
        )

        return out, TaskIntelligenceReport(
            task_summary_available=True,
            task_portfolio_available=False,
            matched_occupations=0,
            unmatched_occupations=len(out),
            warnings=warnings,
            errors=errors,
        )

    out[code_col] = _standardize_code(out[code_col])
    task_summary = task_summary.copy()
    task_summary["O*NET-SOC Code"] = _standardize_code(
        task_summary["O*NET-SOC Code"]
    )

    # Avoid duplicate column names from merge
    summary_cols = [
        col for col in task_summary.columns
        if col != "O*NET-SOC Code"
    ]

    renamed_summary = task_summary.rename(
        columns={col: f"task_{col}" for col in summary_cols}
    )

    merged = out.merge(
        renamed_summary,
        left_on=code_col,
        right_on="O*NET-SOC Code",
        how="left",
    )

    matched = int(merged["O*NET-SOC Code"].notna().sum())
    unmatched = int(merged["O*NET-SOC Code"].isna().sum())

    if unmatched > 0:
        warnings.append(
            f"{unmatched} workforce rows could not be matched to task intelligence."
        )

    return merged, TaskIntelligenceReport(
        task_summary_available=True,
        task_portfolio_available=False,
        matched_occupations=matched,
        unmatched_occupations=unmatched,
        warnings=warnings,
        errors=errors,
    )


def get_top_tasks_for_occupation(
    occupation_code: str,
    task_portfolio: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return top tasks for one occupation.

    Sorting priority:
    1. importance if available
    2. relevance if available
    3. frequency if available
    4. task ID
    """

    portfolio = task_portfolio.copy()
    portfolio["O*NET-SOC Code"] = _standardize_code(
        portfolio["O*NET-SOC Code"]
    )

    occupation_code = str(occupation_code).strip()

    subset = portfolio[
        portfolio["O*NET-SOC Code"] == occupation_code
    ].copy()

    if subset.empty:
        return subset

    sort_candidates = [
        "importance",
        "Importance",
        "task_importance",
        "relevance",
        "Relevance",
        "task_relevance",
        "frequency",
        "Frequency",
        "task_frequency",
    ]

    sort_cols = [col for col in sort_candidates if col in subset.columns]

    if sort_cols:
        subset = subset.sort_values(
            sort_cols,
            ascending=False,
        )
    elif "Task ID" in subset.columns:
        subset = subset.sort_values("Task ID")

    return subset.head(top_n)


def build_role_task_table(
    workforce_df: pd.DataFrame,
    task_portfolio: pd.DataFrame,
    role_col: str = "job_title",
    top_n: int = 5,
) -> pd.DataFrame:
    """Create a compact role-level table with top tasks per occupation."""

    code_col = _find_code_column(workforce_df)

    if code_col is None:
        return pd.DataFrame()

    rows = []

    if role_col not in workforce_df.columns:
        role_col = code_col

    role_occ = (
        workforce_df[[role_col, code_col]]
        .dropna()
        .drop_duplicates()
    )

    for _, row in role_occ.iterrows():
        occupation_code = str(row[code_col]).strip()
        role_name = row[role_col]

        top_tasks = get_top_tasks_for_occupation(
            occupation_code=occupation_code,
            task_portfolio=task_portfolio,
            top_n=top_n,
        )

        if top_tasks.empty:
            rows.append(
                {
                    role_col: role_name,
                    "occupation_code": occupation_code,
                    "top_tasks": "Task portfolio unavailable",
                    "n_top_tasks_returned": 0,
                }
            )
            continue

        task_texts = top_tasks["Task"].astype(str).tolist()

        rows.append(
            {
                role_col: role_name,
                "occupation_code": occupation_code,
                "top_tasks": " | ".join(task_texts),
                "n_top_tasks_returned": len(task_texts),
            }
        )

    return pd.DataFrame(rows)


def attach_task_intelligence(
    workforce_df: pd.DataFrame,
    task_summary_path: str = "data/hcrl_occupation_task_summary.csv",
    task_portfolio_path: str = "data/hcrl_occupation_task_portfolio.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, TaskIntelligenceReport]:
    """Attach task summary and produce role-level top task table.

    Returns:
        enriched_workforce_df
        role_task_table
        report
    """

    warnings: List[str] = []
    errors: List[str] = []

    try:
        task_summary = load_task_summary(task_summary_path)
        task_summary_available = True
    except Exception as e:
        task_summary = pd.DataFrame()
        task_summary_available = False
        errors.append(f"Could not load task summary: {e}")

    try:
        task_portfolio = load_task_portfolio(task_portfolio_path)
        task_portfolio_available = True
    except Exception as e:
        task_portfolio = pd.DataFrame()
        task_portfolio_available = False
        errors.append(f"Could not load task portfolio: {e}")

    if not task_summary_available:
        return workforce_df.copy(), pd.DataFrame(), TaskIntelligenceReport(
            task_summary_available=False,
            task_portfolio_available=task_portfolio_available,
            matched_occupations=0,
            unmatched_occupations=len(workforce_df),
            warnings=warnings,
            errors=errors,
        )

    enriched_df, summary_report = attach_task_summary(
        workforce_df,
        task_summary,
    )

    warnings.extend(summary_report.warnings)
    errors.extend(summary_report.errors)

    if task_portfolio_available:
        role_task_table = build_role_task_table(
            enriched_df,
            task_portfolio,
        )
    else:
        role_task_table = pd.DataFrame()

    return enriched_df, role_task_table, TaskIntelligenceReport(
        task_summary_available=task_summary_available,
        task_portfolio_available=task_portfolio_available,
        matched_occupations=summary_report.matched_occupations,
        unmatched_occupations=summary_report.unmatched_occupations,
        warnings=warnings,
        errors=errors,
    )
