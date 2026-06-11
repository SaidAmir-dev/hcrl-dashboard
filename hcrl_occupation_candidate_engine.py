"""HCRL occupation candidate engine.

Purpose:
Generate higher-quality O*NET occupation candidates before final matching.

This prevents broad fuzzy matching mistakes such as:
- Healthcare Representative -> Materials Scientists
- Laboratory Technician -> Materials Scientists
- Sales Executive -> Chief Executives

This module is generalizable:
It uses function/role cues, not IBM-specific row rules.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


CANDIDATE_TITLE_RULES: Dict[str, List[str]] = {
    "sales": [
        "Sales Managers",
        "Sales Representatives, Wholesale and Manufacturing",
        "Sales Representatives, Wholesale and Manufacturing, Technical and Scientific Products",
        "Sales Representatives, Services",
        "First-Line Supervisors of Retail Sales Workers",
    ],

    "research_science": [
        "Medical Scientists",
        "Biological Scientists",
        "Life Scientists, All Other",
        "Natural Sciences Managers",
        "Clinical Research Coordinators",
        "Chemists",
        "Materials Scientists",
    ],

    "laboratory": [
        "Medical and Clinical Laboratory Technologists",
        "Medical and Clinical Laboratory Technicians",
        "Biological Technicians",
        "Chemical Technicians",
        "Life, Physical, and Social Science Technicians, All Other",
    ],

    "human_resources": [
        "Human Resources Managers",
        "Human Resources Specialists",
        "Compensation, Benefits, and Job Analysis Specialists",
        "Training and Development Specialists",
    ],

    "manufacturing": [
        "Industrial Production Managers",
        "Production, Planning, and Expediting Clerks",
        "First-Line Supervisors of Production and Operating Workers",
        "Industrial Engineers",
        "Manufacturing Engineers",
    ],

    "healthcare": [
        "Medical and Health Services Managers",
        "Patient Representatives",
        "Healthcare Social Workers",
        "Health Information Technologists and Medical Registrars",
        "Medical Secretaries and Administrative Assistants",
    ],

    "operations": [
        "General and Operations Managers",
        "Operations Research Analysts",
        "Business Operations Specialists, All Other",
        "Project Management Specialists",
    ],

    "software_it": [
        "Software Developers",
        "Computer and Information Systems Managers",
        "Data Scientists",
        "Computer Systems Analysts",
        "Database Architects",
        "Web Developers",
    ],

    "finance": [
        "Financial Managers",
        "Financial Analysts",
        "Accountants and Auditors",
        "Budget Analysts",
        "Treasurers and Controllers",
    ],

    "marketing": [
        "Marketing Managers",
        "Market Research Analysts and Marketing Specialists",
        "Advertising and Promotions Managers",
    ],

    "customer_service": [
        "Customer Service Representatives",
        "First-Line Supervisors of Office and Administrative Support Workers",
        "Information and Record Clerks, All Other",
    ],
}


TITLE_CUE_RULES: Dict[str, List[str]] = {
    "laboratory": [
        "laboratory", "lab", "technician", "clinical laboratory"
    ],
    "healthcare": [
        "healthcare", "health care", "patient", "medical representative", "clinical"
    ],
    "research_science": [
        "research", "scientist", "r&d", "science"
    ],
    "sales": [
        "sales", "account executive", "sales representative", "business development"
    ],
    "human_resources": [
        "human resources", "hr", "talent", "recruit"
    ],
    "manufacturing": [
        "manufacturing", "production", "plant", "industrial"
    ],
    "operations": [
        "operations", "general manager", "manager"
    ],
    "software_it": [
        "software", "developer", "data", "programmer", "it"
    ],
    "finance": [
        "finance", "financial", "accounting", "accountant"
    ],
    "marketing": [
        "marketing", "brand", "advertising"
    ],
    "customer_service": [
        "customer service", "customer support", "client service"
    ],
}


def _clean(x: object) -> str:
    return (
        str(x)
        .lower()
        .strip()
        .replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
        .replace("&", "and")
    )


def infer_candidate_family(
    job_title: object,
    detected_function: Optional[str] = None,
    department: object = None,
) -> Optional[str]:

    text = _clean(f"{job_title} {department or ''}")

    # Specific cues first
    for family, cues in TITLE_CUE_RULES.items():
        if any(_clean(cue) in text for cue in cues):
            return family

    # Fallback to normalized function
    if detected_function in CANDIDATE_TITLE_RULES:
        return detected_function

    return None


def generate_candidate_titles(
    job_title: object,
    detected_function: Optional[str] = None,
    department: object = None,
) -> List[str]:

    family = infer_candidate_family(
        job_title=job_title,
        detected_function=detected_function,
        department=department,
    )

    if family is None:
        return []

    return CANDIDATE_TITLE_RULES.get(family, [])


def filter_onet_reference_to_candidates(
    onet_reference: pd.DataFrame,
    candidate_titles: List[str],
    title_col: str = "Title",
) -> pd.DataFrame:

    if not candidate_titles:
        return onet_reference

    ref = onet_reference.copy()

    candidate_clean = {_clean(title) for title in candidate_titles}

    filtered = ref[
        ref[title_col].astype(str).apply(
            lambda title: _clean(title) in candidate_clean
        )
    ].copy()

    if filtered.empty:
        return onet_reference

    return filtered
