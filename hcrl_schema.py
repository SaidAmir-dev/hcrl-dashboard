"""HCRL universal workforce schema layer.

This module converts source-specific workforce files into a stable HCRL data contract.

Core principle:
HCRL must not depend on IBM, Workday, SAP, ADP, or any single company's column names.

The schema layer supports:
1. exact column matching
2. normalized matching
3. token-order-insensitive matching

Example:
    Monthly_Income
    monthly income
    income_monthly
    Monthly-Income

can all be mapped safely when the meaning is clear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


HCRL_CANONICAL_FIELDS: Dict[str, List[str]] = {
    "employee_id": [
        "employee_id", "employee id", "employeeid", "employee_number",
        "employee number", "employeenumber", "EmployeeNumber",
        "EmployeeID", "worker_id", "worker id", "staff_id",
        "staff id", "id"
    ],

    "job_title": [
        "job_title", "job title", "jobrole", "JobRole",
        "job_role", "role", "position", "position_title",
        "position title", "title", "Title", "occupation",
        "Occupation", "occupation_title", "occupation title"
    ],

    "occupation_code": [
        "occupation_code", "occupation code", "soc_code", "soc code",
        "SOC", "SOC Code", "onet_soc_code", "onet soc code",
        "O*NET-SOC Code", "onet_code", "onet code"
    ],

    "department": [
        "department", "Department", "business_unit", "business unit",
        "BusinessUnit", "division", "team", "unit", "function",
        "org_unit", "organization_unit"
    ],

    "location": [
        "location", "Location", "state", "State", "region",
        "Region", "city", "City", "country", "Country",
        "office", "site", "work_location", "work location"
    ],

    "annual_wage": [
        "annual_wage", "annual wage", "annual_salary", "annual salary",
        "AnnualSalary", "salary", "Salary", "annual_wage_proxy",
        "annual wage proxy", "annual_compensation", "annual compensation",
        "compensation", "base_salary", "base salary", "yearly_salary",
        "yearly salary", "annual_pay", "annual pay", "gross_annual_pay"
    ],

    "monthly_income": [
        "MonthlyIncome", "monthly_income", "monthly income",
        "Monthly_Income", "income_monthly", "Income_Monthly",
        "monthly_salary", "monthly salary", "MonthlySalary",
        "monthly_compensation", "monthly compensation",
        "Monthly_Compensation", "monthly_pay", "monthly pay",
        "MonthlyPay", "monthly_earnings", "monthly earnings",
        "Monthly_Earnings"
    ],

    "tenure_years": [
        "tenure_years", "tenure years", "YearsAtCompany",
        "years_at_company", "years at company", "tenure",
        "company_tenure", "company tenure", "service_years",
        "service years", "years_service", "years of service"
    ],

    "separation_outcome": [
        "separation_outcome", "separation outcome", "attrition_target",
        "attrition target", "Attrition", "attrition", "separated",
        "turnover", "left_company", "left company", "exit",
        "employee_exit", "employee exit", "termination",
        "resigned"
    ],

    "predicted_attrition_probability": [
        "predicted_attrition_probability",
        "predicted attrition probability",
        "predicted_risk", "predicted risk",
        "attrition_probability", "attrition probability",
        "turnover_probability", "turnover probability",
        "separation_probability", "separation probability"
    ],

    "replacement_cost_multiplier": [
        "replacement_cost_multiplier", "replacement cost multiplier",
        "replacement_multiplier", "replacement multiplier",
        "lambda", "cost_multiplier", "cost multiplier"
    ],

    "replacement_cost": [
        "replacement_cost", "replacement cost",
        "replacement_cost_estimate", "replacement cost estimate",
        "turnover_cost", "turnover cost", "separation_cost",
        "separation cost"
    ],
}


IBM_FEATURES: List[str] = [
    "Age", "BusinessTravel", "Department", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "Gender", "JobInvolvement", "JobLevel",
    "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome",
    "NumCompaniesWorked", "OverTime", "PercentSalaryHike",
    "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]


MODEL_EXCLUDE_FIELDS = {
    "employee_id",
    "separation_outcome",
    "predicted_attrition_probability",
    "replacement_cost",
    "replacement_cost_multiplier",
}


@dataclass
class SchemaReport:
    source_type: str
    mapped_columns: Dict[str, str]
    warnings: List[str]
    errors: List[str]
    model_feature_columns: List[str]
    unmapped_columns: List[str]

    @property
    def has_company_attrition_history(self) -> bool:
        return "separation_outcome" in self.mapped_columns

    @property
    def has_precomputed_risk(self) -> bool:
        return "predicted_attrition_probability" in self.mapped_columns

    @property
    def can_train_company_risk_model(self) -> bool:
        return self.has_company_attrition_history and len(self.model_feature_columns) > 0

    @property
    def can_use_external_baseline_risk(self) -> bool:
        return "job_title" in self.mapped_columns or "occupation_code" in self.mapped_columns

    @property
    def is_valid_for_cost(self) -> bool:
        return "annual_wage" in self.mapped_columns or "monthly_income" in self.mapped_columns

    @property
    def is_valid_for_onet(self) -> bool:
        return "job_title" in self.mapped_columns or "occupation_code" in self.mapped_columns


def _normalize_name(x: str) -> str:
    """Normalize column names by removing separators and casing differences."""
    return (
        str(x)
        .lower()
        .strip()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
        .replace("/", "")
        .replace("\\", "")
        .replace("(", "")
        .replace(")", "")
    )


def _token_signature(x: str) -> str:
    """Create token-order-insensitive signature.

    Example:
        monthly_income -> income_monthly
        income_monthly -> income_monthly
    """
    cleaned = (
        str(x)
        .lower()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "_")
        .replace(")", "_")
    )
    tokens = [t for t in cleaned.split("_") if t]
    return "_".join(sorted(tokens))


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching source column using exact, normalized, and token matching."""

    # 1. Exact match
    exact_columns = set(df.columns)

    for candidate in candidates:
        if candidate in exact_columns:
            return candidate

    # 2. Normalized match
    normalized_lookup = {_normalize_name(col): col for col in df.columns}

    for candidate in candidates:
        key = _normalize_name(candidate)
        if key in normalized_lookup:
            return normalized_lookup[key]

    # 3. Token-order-insensitive match
    token_lookup = {_token_signature(col): col for col in df.columns}

    for candidate in candidates:
        key = _token_signature(candidate)
        if key in token_lookup:
            return token_lookup[key]

    return None


def infer_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """Infer mapping from company-specific columns to HCRL canonical fields."""
    mapping: Dict[str, str] = {}

    for canonical_field, candidate_names in HCRL_CANONICAL_FIELDS.items():
        matched_column = _first_present(df, candidate_names)

        if matched_column is not None:
            mapping[canonical_field] = matched_column

    return mapping


def _standardize_separation_outcome(series: pd.Series) -> pd.Series:
    """Convert common attrition / separation labels into 0/1."""

    outcome_map = {
        "Yes": 1,
        "yes": 1,
        "YES": 1,
        "Y": 1,
        "y": 1,
        "True": 1,
        "true": 1,
        "TRUE": 1,
        True: 1,
        "1": 1,
        1: 1,

        "No": 0,
        "no": 0,
        "NO": 0,
        "N": 0,
        "n": 0,
        "False": 0,
        "false": 0,
        "FALSE": 0,
        False: 0,
        "0": 0,
        0: 0,
    }

    return series.map(outcome_map).fillna(series).pipe(
        pd.to_numeric, errors="coerce"
    )


def _append_canonical_columns(
    df: pd.DataFrame,
    mapping: Dict[str, str]
) -> pd.DataFrame:
    """Append canonical HCRL columns while preserving original company columns."""
    out = df.copy()

    for canonical_field, original_column in mapping.items():
        if canonical_field not in out.columns:
            out[canonical_field] = out[original_column]

    return out


def _detect_source_type(df: pd.DataFrame) -> str:
    """Detect known data sources without making the whole system dependent on them."""
    if "Attrition" in df.columns and "MonthlyIncome" in df.columns:
        return "ibm_hr_attrition_dataset"

    return "generic_company_workforce_file"


def _build_model_feature_columns(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    source_type: str
) -> List[str]:
    """Identify usable model features without including IDs, outcomes, or outputs."""

    if source_type == "ibm_hr_attrition_dataset":
        return [col for col in IBM_FEATURES if col in df.columns]

    excluded_columns = set(mapping.values()) | MODEL_EXCLUDE_FIELDS
    excluded_columns.update(HCRL_CANONICAL_FIELDS.keys())

    model_features: List[str] = []

    for col in df.columns:
        col_str = str(col).lower()

        if col in excluded_columns:
            continue

        if col_str.endswith("date"):
            continue

        if "date" in col_str and ("hire" in col_str or "exit" in col_str or "termination" in col_str):
            continue

        if df[col].nunique(dropna=True) <= 1:
            continue

        model_features.append(col)

    return model_features


def standardize_workforce_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, SchemaReport]:
    """Standardize uploaded workforce data into HCRL canonical structure.

    Returns:
        standardized_dataframe, schema_report
    """

    if df.empty:
        report = SchemaReport(
            source_type="empty_file",
            mapped_columns={},
            warnings=[],
            errors=["Uploaded dataset is empty."],
            model_feature_columns=[],
            unmapped_columns=[],
        )
        return df.copy(), report

    mapping = infer_column_mapping(df)
    source_type = _detect_source_type(df)

    warnings: List[str] = []
    errors: List[str] = []

    if source_type == "ibm_hr_attrition_dataset":
        warnings.append(
            "IBM HR dataset detected. IBM is treated as a source adapter, "
            "not as the enterprise HCRL schema."
        )

    out = _append_canonical_columns(df, mapping)

    # Standardize separation outcome if available
    if "separation_outcome" in out.columns:
        out["separation_outcome"] = _standardize_separation_outcome(
            out["separation_outcome"]
        )

        valid_outcomes = out["separation_outcome"].dropna().isin([0, 1]).all()

        if not valid_outcomes:
            warnings.append(
                "Separation outcome contains values outside 0/1 after conversion. "
                "Company-specific attrition modeling may be unreliable."
            )

    # Derive annual wage from monthly income if annual wage is unavailable
    if "annual_wage" not in out.columns and "monthly_income" in out.columns:
        out["monthly_income"] = pd.to_numeric(out["monthly_income"], errors="coerce")
        out["annual_wage"] = out["monthly_income"] * 12
        mapping["annual_wage"] = "derived_from_monthly_income"

    # Validate wage
    if "annual_wage" in out.columns:
        out["annual_wage"] = pd.to_numeric(out["annual_wage"], errors="coerce")

        if out["annual_wage"].notna().sum() == 0:
            errors.append(
                "Annual wage field exists, but contains no usable numeric values."
            )

        if out["annual_wage"].dropna().lt(0).any():
            errors.append("Annual wage cannot contain negative values.")

    # Validate predicted risk if supplied
    if "predicted_attrition_probability" in out.columns:
        out["predicted_attrition_probability"] = pd.to_numeric(
            out["predicted_attrition_probability"], errors="coerce"
        )

        invalid_probability = (
            out["predicted_attrition_probability"]
            .dropna()
            .pipe(lambda s: s.lt(0).any() or s.gt(1).any())
        )

        if invalid_probability:
            errors.append(
                "Predicted attrition probabilities must be between 0 and 1."
            )

    # Enterprise-safe warnings
    if "employee_id" not in out.columns:
        warnings.append(
            "No employee ID detected. HCRL can still analyze workforce segments, "
            "but individual-level traceability will be limited."
        )

    if "job_title" not in out.columns and "occupation_code" not in out.columns:
        warnings.append(
            "No job title or occupation code detected. O*NET task intelligence "
            "and AI exposure mapping will be unavailable."
        )

    if "annual_wage" not in out.columns:
        warnings.append(
            "No wage or compensation field detected. Economic exposure can only "
            "be estimated after compensation data or external wage benchmarks are added."
        )

    if "separation_outcome" not in out.columns:
        warnings.append(
            "No historical separation outcome detected. HCRL cannot train a "
            "company-specific attrition model, but it can still use external "
            "labor-market baseline risk models."
        )

    model_feature_columns = _build_model_feature_columns(
        out,
        mapping,
        source_type,
    )

    if "separation_outcome" in out.columns and len(model_feature_columns) == 0:
        warnings.append(
            "Separation outcome exists, but no usable model feature columns were detected. "
            "Company-specific risk modeling will be limited."
        )

    mapped_original_columns = set(mapping.values())
    unmapped_columns = [
        col for col in df.columns
        if col not in mapped_original_columns
    ]

    report = SchemaReport(
        source_type=source_type,
        mapped_columns=mapping,
        warnings=warnings,
        errors=errors,
        model_feature_columns=model_feature_columns,
        unmapped_columns=unmapped_columns,
    )

    return out, report
