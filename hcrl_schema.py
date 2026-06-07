"""HCRL universal workforce schema layer.

This module converts source-specific workforce files into a stable HCRL data contract.
It deliberately avoids silently fabricating missing fields. If a field is unavailable,
it is either left missing with a validation warning or the workflow is stopped by the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


HCRL_CANONICAL_FIELDS: Dict[str, List[str]] = {
    "employee_id": ["employee_id", "EmployeeNumber", "EmployeeID", "employee_number", "id"],
    "job_title": ["job_title", "JobRole", "job_role", "Title", "occupation", "Occupation", "position_title"],
    "occupation_code": ["occupation_code", "soc_code", "SOC", "O*NET-SOC Code", "onet_soc_code"],
    "department": ["department", "Department", "business_unit", "BusinessUnit", "division"],
    "location": ["location", "Location", "state", "State", "region", "Region", "city", "City"],
    "annual_wage": ["annual_wage", "annual_salary", "salary", "AnnualSalary", "annual_wage_proxy"],
    "monthly_income": ["MonthlyIncome", "monthly_income", "monthly_salary"],
    "tenure_years": ["tenure_years", "YearsAtCompany", "tenure", "company_tenure"],
    "separation_outcome": ["separation_outcome", "attrition_target", "Attrition", "separated", "turnover"],
    "predicted_attrition_probability": ["predicted_attrition_probability", "predicted_risk", "attrition_probability"],
    "replacement_cost_multiplier": ["replacement_cost_multiplier", "lambda", "replacement_multiplier"],
    "replacement_cost": ["replacement_cost", "replacement_cost_estimate"],
}

IBM_FEATURES: List[str] = [
    "Age", "BusinessTravel", "Department", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "Gender", "JobInvolvement", "JobLevel", "JobRole",
    "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "NumCompaniesWorked",
    "OverTime", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

MODEL_EXCLUDE_FIELDS = {
    "employee_id", "separation_outcome", "predicted_attrition_probability",
    "replacement_cost", "replacement_cost_multiplier"
}


@dataclass
class SchemaReport:
    source_type: str
    mapped_columns: Dict[str, str]
    warnings: List[str]
    errors: List[str]
    model_feature_columns: List[str]

    @property
    def is_valid_for_risk(self) -> bool:
        has_prediction = "predicted_attrition_probability" in self.mapped_columns
        has_outcome = "separation_outcome" in self.mapped_columns
        return has_prediction or has_outcome

    @property
    def is_valid_for_cost(self) -> bool:
        has_wage = "annual_wage" in self.mapped_columns or "monthly_income" in self.mapped_columns
        has_cost = "replacement_cost" in self.mapped_columns or "replacement_cost_multiplier" in self.mapped_columns
        return has_wage and has_cost


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    lower_lookup = {str(c).lower().strip(): c for c in df.columns}
    for col in candidates:
        key = col.lower().strip()
        if key in lower_lookup:
            return lower_lookup[key]
    return None


def infer_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for canonical, candidates in HCRL_CANONICAL_FIELDS.items():
        found = _first_present(df, candidates)
        if found is not None:
            mapping[canonical] = found
    return mapping


def standardize_workforce_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, SchemaReport]:
    """Return a copy with canonical HCRL fields appended where possible."""
    out = df.copy()
    mapping = infer_column_mapping(out)
    warnings: List[str] = []
    errors: List[str] = []

    source_type = "generic_company_workforce_file"
    if "Attrition" in out.columns and "MonthlyIncome" in out.columns:
        source_type = "ibm_hr_attrition_dataset"
        warnings.append(
            "IBM HR dataset detected. IBM is treated as a source adapter, not as the enterprise schema."
        )

    for canonical, original in mapping.items():
        if canonical not in out.columns:
            out[canonical] = out[original]

    if "separation_outcome" in out.columns:
        if out["separation_outcome"].dtype == object:
            out["separation_outcome"] = out["separation_outcome"].map(
                {"Yes": 1, "No": 0, "yes": 1, "no": 0, "Y": 1, "N": 0, "true": 1, "false": 0, "True": 1, "False": 0}
            ).fillna(out["separation_outcome"])
        out["separation_outcome"] = pd.to_numeric(out["separation_outcome"], errors="coerce")

    if "annual_wage" not in out.columns and "monthly_income" in out.columns:
        out["annual_wage"] = pd.to_numeric(out["monthly_income"], errors="coerce") * 12
        mapping["annual_wage"] = "derived_from_monthly_income"

    if "predicted_attrition_probability" in out.columns:
        out["predicted_attrition_probability"] = pd.to_numeric(
            out["predicted_attrition_probability"], errors="coerce"
        )
        invalid = out["predicted_attrition_probability"].dropna().lt(0).sum() + out["predicted_attrition_probability"].dropna().gt(1).sum()
        if invalid:
            errors.append("Predicted attrition probabilities must be between 0 and 1.")

    if "annual_wage" not in out.columns:
        errors.append("No annual wage or monthly income field detected. Cost exposure cannot be estimated.")

    if "job_title" not in out.columns and "occupation_code" not in out.columns:
        warnings.append("No job title or occupation code detected. O*NET matching will be unavailable.")

    if "separation_outcome" not in out.columns and "predicted_attrition_probability" not in out.columns:
        errors.append(
            "No separation outcome or predicted attrition probability detected. Risk engine cannot estimate attrition risk."
        )

    excluded_originals = set(mapping.values()) | MODEL_EXCLUDE_FIELDS
    excluded_originals.update(["Attrition"])
    model_feature_columns: List[str]
    if source_type == "ibm_hr_attrition_dataset":
        model_feature_columns = [c for c in IBM_FEATURES if c in out.columns]
    else:
        model_feature_columns = [
            c for c in out.columns
            if c not in excluded_originals
            and not str(c).lower().endswith("date")
            and out[c].nunique(dropna=True) > 1
        ]

    if "separation_outcome" in out.columns and len(model_feature_columns) == 0:
        errors.append("Separation outcome exists, but no usable model feature columns were detected.")

    report = SchemaReport(
        source_type=source_type,
        mapped_columns=mapping,
        warnings=warnings,
        errors=errors,
        model_feature_columns=model_feature_columns,
    )
    return out, report
