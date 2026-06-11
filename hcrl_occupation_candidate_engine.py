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
        "Biochemists and Biophysicists",
        "Life Scientists, All Other",
        "Clinical Research Coordinators",
        "Natural Sciences Managers",
        "Chemists",
        "Materials Scientists",
    ],

    "laboratory": [
        "Medical and Clinical Laboratory Technologists",
        "Medical and Clinical Laboratory Technicians",
        "Clinical Laboratory Technologists and Technicians",
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
        "First-Line Supervisors of Production and Operating Workers",
        "Manufacturing Engineers",
        "Industrial Engineers",
        "Production, Planning, and Expediting Clerks",
        "General and Operations Managers",
    ],

    "healthcare": [
        "Medical and Health Services Managers",
        "Patient Representatives",
        "Healthcare Social Workers",
        "Health Information Technologists and Medical Registrars",
        "Medical Secretaries and Administrative Assistants",
        "Customer Service Representatives",
    ],

    "operations": [
        "General and Operations Managers",
        "Operations Research Analysts",
        "Business Operations Specialists, All Other",
        "Project Management Specialists",
        "First-Line Supervisors of Office and Administrative Support Workers",
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
        "laboratory",
        "lab",
        "lab technician",
        "laboratory technician",
        "clinical laboratory",
        "medical laboratory",
        "technician",
    ],

    "healthcare": [
        "healthcare",
        "health care",
        "patient",
        "medical representative",
        "health representative",
        "clinical",
    ],

    "research_science": [
        "research scientist",
        "research director",
        "principal scientist",
        "senior scientist",
        "scientist",
        "research",
        "r&d",
        "science",
    ],

    "sales": [
        "sales",
        "account executive",
        "sales executive",
        "sales representative",
        "business development",
        "account manager",
    ],

    "human_resources": [
        "human resources",
        "hr",
        "talent",
        "recruit",
        "people operations",
    ],

    "manufacturing": [
        "manufacturing director",
        "manufacturing manager",
        "manufacturing",
        "production director",
        "production manager",
        "production",
        "plant manager",
        "industrial",
    ],

    "operations": [
        "operations",
        "general manager",
        "business operations",
        "manager",
    ],

    "software_it": [
        "software",
        "developer",
        "data",
        "programmer",
        "it",
        "information technology",
    ],

    "finance": [
        "finance",
        "financial",
        "accounting",
        "accountant",
    ],

    "marketing": [
        "marketing",
        "brand",
        "advertising",
    ],

    "customer_service": [
        "customer service",
        "customer support",
        "client service",
        "support",
    ],
}


SPECIFIC_TITLE_CANDIDATES = {
    "sales executive": [
        "Sales Managers",
        "Sales Representatives, Wholesale and Manufacturing, Except Technical and Scientific Products",
        "Sales Representatives, Wholesale and Manufacturing, Technical and Scientific Products",
    ],

    "sales representative": [
        "Sales Representatives, Wholesale and Manufacturing, Except Technical and Scientific Products",
        "Sales Representatives, Wholesale and Manufacturing, Technical and Scientific Products",
    ],

    "research scientist": [
        "Medical Scientists, Except Epidemiologists",
        "Biological Scientists, All Other",
        "Biochemists and Biophysicists",
        "Chemists",
        "Natural Sciences Managers",
    ],

    "laboratory technician": [
        "Clinical Laboratory Technologists and Technicians",
        "Biological Technicians",
        "Chemical Technicians",
        "Life, Physical, and Social Science Technicians, All Other",
    ],

    "manufacturing director": [
        "Industrial Production Managers",
        "First-Line Supervisors of Production and Operating Workers",
        "Industrial Engineers",
        "General and Operations Managers",
    ],

    "healthcare representative": [
        "Patient Representatives",
        "Customer Service Representatives",
        "Medical Secretaries and Administrative Assistants",
        "Health Information Technologists and Medical Registrars",
    ],

    "human resources": [
        "Human Resources Specialists",
        "Human Resources Managers",
    ],

    "manager": [
        "General and Operations Managers",
        "Sales Managers",
        "Human Resources Managers",
        "Industrial Production Managers",
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

    for family, cues in TITLE_CUE_RULES.items():
        if any(_clean(cue) in text for cue in cues):
            return family

    if detected_function in CANDIDATE_TITLE_RULES:
        return detected_function

    return None


def generate_candidate_titles(
    job_title: object,
    detected_function: Optional[str] = None,
    department: object = None,
) -> List[str]:

    text = _clean(job_title)

    for specific_title, candidates in SPECIFIC_TITLE_CANDIDATES.items():
        if _clean(specific_title) in text:
            return candidates

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
