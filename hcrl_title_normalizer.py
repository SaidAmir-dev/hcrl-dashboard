"""HCRL title normalization engine.

Purpose:
Convert messy company job titles into cleaner canonical occupation titles
before O*NET matching.

This is not IBM-specific.
It uses general title parsing:
- function detection
- seniority detection
- canonical occupation candidates

The output is still auditable and conservative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class NormalizedTitle:
    original_title: str
    detected_function: Optional[str]
    detected_level: Optional[str]
    canonical_title: Optional[str]
    normalization_method: str


FUNCTION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "sales": (
        "sales",
        "account executive",
        "account manager",
        "business development",
        "revenue",
    ),
    "research_science": (
        "research",
        "scientist",
        "laboratory",
        "lab",
        "r&d",
        "clinical research",
    ),
    "human_resources": (
        "human resources",
        "hr",
        "people operations",
        "talent",
        "recruiting",
        "recruiter",
    ),
    "manufacturing": (
        "manufacturing",
        "production",
        "plant",
        "industrial",
        "factory",
    ),
    "healthcare": (
        "healthcare",
        "health care",
        "medical",
        "clinical",
        "patient",
    ),
    "operations": (
        "operations",
        "general manager",
        "business operations",
        "ops",
    ),
    "software_it": (
        "software",
        "developer",
        "programmer",
        "engineer",
        "data",
        "information technology",
        "it ",
    ),
    "finance": (
        "finance",
        "financial",
        "accounting",
        "accountant",
        "controller",
        "budget",
    ),
    "marketing": (
        "marketing",
        "brand",
        "advertising",
        "promotion",
        "market research",
    ),
    "customer_service": (
        "customer service",
        "customer support",
        "client service",
        "client success",
        "support representative",
    ),
}


LEVEL_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "executive": (
        "chief",
        "ceo",
        "cfo",
        "coo",
        "cto",
        "president",
        "vice president",
        "vp",
    ),
    "director": (
        "director",
        "head",
        "executive",
    ),
    "manager": (
        "manager",
        "supervisor",
    ),
    "senior_individual_contributor": (
        "senior",
        "sr",
        "principal",
        "lead",
        "staff",
    ),
    "individual_contributor": (
        "representative",
        "specialist",
        "scientist",
        "technician",
        "analyst",
        "associate",
        "coordinator",
    ),
}


CANONICAL_TITLE_RULES: Dict[Tuple[str, str], str] = {
    ("sales", "executive"): "Sales Managers",
    ("sales", "director"): "Sales Managers",
    ("sales", "manager"): "Sales Managers",
    ("sales", "individual_contributor"): "Sales Representatives, Wholesale and Manufacturing",

    ("research_science", "director"): "Natural Sciences Managers",
    ("research_science", "manager"): "Natural Sciences Managers",
    ("research_science", "senior_individual_contributor"): "Medical Scientists",
    ("research_science", "individual_contributor"): "Medical Scientists",

    ("human_resources", "director"): "Human Resources Managers",
    ("human_resources", "manager"): "Human Resources Managers",
    ("human_resources", "individual_contributor"): "Human Resources Specialists",

    ("manufacturing", "director"): "Industrial Production Managers",
    ("manufacturing", "manager"): "Industrial Production Managers",
    ("manufacturing", "individual_contributor"): "Production Workers, All Other",

    ("healthcare", "director"): "Medical and Health Services Managers",
    ("healthcare", "manager"): "Medical and Health Services Managers",
    ("healthcare", "individual_contributor"): "Patient Representatives",

    ("operations", "executive"): "General and Operations Managers",
    ("operations", "director"): "General and Operations Managers",
    ("operations", "manager"): "General and Operations Managers",

    ("software_it", "director"): "Computer and Information Systems Managers",
    ("software_it", "manager"): "Computer and Information Systems Managers",
    ("software_it", "senior_individual_contributor"): "Software Developers",
    ("software_it", "individual_contributor"): "Software Developers",

    ("finance", "director"): "Financial Managers",
    ("finance", "manager"): "Financial Managers",
    ("finance", "individual_contributor"): "Financial Analysts",

    ("marketing", "director"): "Marketing Managers",
    ("marketing", "manager"): "Marketing Managers",
    ("marketing", "individual_contributor"): "Market Research Analysts and Marketing Specialists",

    ("customer_service", "manager"): "Customer Service Representatives",
    ("customer_service", "individual_contributor"): "Customer Service Representatives",
}


def _clean_text(x: object) -> str:
    return (
        str(x)
        .lower()
        .strip()
        .replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
        .replace("&", "and")
    )


def detect_function(title: object, department: object = None) -> Optional[str]:
    text = _clean_text(f"{title} {department or ''}")

    best_function = None
    best_score = 0

    for function, keywords in FUNCTION_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text)

        if score > best_score:
            best_score = score
            best_function = function

    return best_function if best_score > 0 else None


def detect_level(title: object) -> Optional[str]:
    text = _clean_text(title)

    for level, keywords in LEVEL_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return level

    return None


def normalize_title(title: object, department: object = None) -> NormalizedTitle:
    original_title = str(title)

    detected_function = detect_function(title, department)
    detected_level = detect_level(title)

    canonical_title = None
    method = "no_normalization_rule_matched"

    if detected_function and detected_level:
        canonical_title = CANONICAL_TITLE_RULES.get(
            (detected_function, detected_level)
        )

        if canonical_title:
            method = "function_level_rule"

    return NormalizedTitle(
        original_title=original_title,
        detected_function=detected_function,
        detected_level=detected_level,
        canonical_title=canonical_title,
        normalization_method=method,
    )
