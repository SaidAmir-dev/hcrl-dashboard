"""HCRL O*NET occupation resolver.

Generalized occupation mapping for any company.

Principle:
Do NOT rely on raw fuzzy title matching alone.

Use:
1. exact O*NET-SOC code match
2. exact title match
3. title normalization
4. occupational family detection
5. family-constrained fuzzy matching
6. confidence/status audit layer
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import pandas as pd

from hcrl_title_normalizer import normalize_title


AUTO_ACCEPT_THRESHOLD = 0.88
REVIEW_THRESHOLD = 0.70


@dataclass
class OnetMappingReport:
    role_column: Optional[str]
    coverage: float
    accepted_matches: int
    review_required_matches: int
    unmatched: int
    exact_matches: int
    fuzzy_matches: int
    note: str


OCCUPATION_FAMILY_KEYWORDS: Dict[str, List[str]] = {
    "sales": [
        "sales", "account executive", "account manager", "business development",
        "sales representative", "sales executive", "sales manager"
    ],
    "software_it": [
        "software", "developer", "engineer", "programmer", "data engineer",
        "backend", "frontend", "full stack", "systems", "information systems"
    ],
    "research_science": [
        "research", "scientist", "laboratory", "lab", "clinical research",
        "r&d", "biology", "chemist", "medical scientist"
    ],
    "human_resources": [
        "human resources", "hr", "talent", "recruiter", "people operations",
        "compensation", "benefits"
    ],
    "healthcare": [
        "healthcare", "health care", "medical", "patient", "clinical",
        "nurse", "physician", "health services", "hospital"
    ],
    "manufacturing": [
        "manufacturing", "production", "plant", "industrial", "factory",
        "quality", "supply chain"
    ],
    "operations": [
        "operations", "general manager", "business operations", "ops"
    ],
    "finance": [
        "finance", "financial", "accounting", "accountant", "analyst",
        "controller", "treasurer", "investment"
    ],
    "marketing": [
        "marketing", "brand", "advertising", "promotion", "market research"
    ],
    "customer_service": [
        "customer service", "customer support", "client service",
        "client success", "support representative"
    ],
}


ONET_FAMILY_FILTERS: Dict[str, List[str]] = {
    "sales": [
        "sales", "account", "business development", "marketing"
    ],
    "software_it": [
        "software", "developer", "programmer", "computer",
        "information systems", "data", "web", "network", "database"
    ],
    "research_science": [
        "scientist", "research", "laboratory", "clinical", "biological",
        "chemist", "medical scientists", "natural sciences"
    ],
    "human_resources": [
        "human resources", "training", "compensation", "benefits", "recruit"
    ],
    "healthcare": [
        "medical", "health", "patient", "clinical", "nurse", "physician"
    ],
    "manufacturing": [
        "production", "manufacturing", "industrial", "quality",
        "operations", "supply chain", "logistics"
    ],
    "operations": [
        "operations", "general and operations", "business operations"
    ],
    "finance": [
        "financial", "accountants", "auditors", "budget", "investment",
        "treasurers", "controllers"
    ],
    "marketing": [
        "marketing", "advertising", "promotions", "market research"
    ],
    "customer_service": [
        "customer", "service", "representatives", "client"
    ],
}


def _clean(x) -> str:
    return (
        str(x)
        .lower()
        .strip()
        .replace("&", "and")
        .replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
        .replace(",", " ")
        .replace(".", " ")
    )


def _compact(x) -> str:
    return "".join(_clean(x).split())


def _token_set(x: str) -> set:
    return set(_clean(x).split())


def _token_overlap_score(a: str, b: str) -> float:
    a_tokens = _token_set(a)
    b_tokens = _token_set(b)

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _combined_similarity(a: str, b: str) -> float:
    seq_score = SequenceMatcher(None, _clean(a), _clean(b)).ratio()
    compact_score = SequenceMatcher(None, _compact(a), _compact(b)).ratio()
    token_score = _token_overlap_score(a, b)
    return max(seq_score, compact_score, token_score)


def _detect_role_column(df: pd.DataFrame) -> Optional[str]:
    for col in [
        "occupation_code",
        "matched_onet_code",
        "soc_code",
        "job_title",
        "JobRole",
        "job_role",
        "Title",
        "occupation",
        "Occupation",
        "position",
        "role",
    ]:
        if col in df.columns:
            return col
    return None


def _prepare_reference(onet_reference: pd.DataFrame) -> Tuple[pd.DataFrame, str, Optional[str]]:
    ref = onet_reference.copy()

    title_col = "Title" if "Title" in ref.columns else None
    code_col = "O*NET-SOC Code" if "O*NET-SOC Code" in ref.columns else None

    if title_col is None:
        raise ValueError("O*NET reference must contain a Title column.")

    ref["_clean_title"] = ref[title_col].apply(_clean)
    ref["_compact_title"] = ref[title_col].apply(_compact)

    if code_col is not None:
        ref[code_col] = ref[code_col].astype(str).str.strip()

    return ref, title_col, code_col


def _infer_family(job_title: str, department: Optional[str] = None) -> Optional[str]:
    combined = _clean(f"{job_title} {department or ''}")

    best_family = None
    best_hits = 0

    for family, keywords in OCCUPATION_FAMILY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if _clean(kw) in combined)

        if hits > best_hits:
            best_hits = hits
            best_family = family

    return best_family if best_hits > 0 else None


def _filter_reference_by_family(
    ref: pd.DataFrame,
    title_col: str,
    family: Optional[str],
) -> pd.DataFrame:
    if family is None or family not in ONET_FAMILY_FILTERS:
        return ref

    keywords = ONET_FAMILY_FILTERS[family]

    mask = ref[title_col].astype(str).apply(
        lambda title: any(_clean(kw) in _clean(title) for kw in keywords)
    )

    filtered = ref[mask].copy()

    if filtered.empty:
        return ref

    return filtered


def _exact_code_match(
    value: str,
    ref: pd.DataFrame,
    code_col: Optional[str],
) -> Optional[pd.Series]:

    if code_col is None:
        return None

    value_str = str(value).strip()
    matches = ref[ref[code_col].astype(str).str.strip() == value_str]

    if matches.empty:
        return None

    return matches.iloc[0]


def _exact_title_match(
    value: str,
    ref: pd.DataFrame,
    title_col: str,
) -> Optional[pd.Series]:

    value_clean = _clean(value)
    value_compact = _compact(value)

    matches = ref[
        (ref["_clean_title"] == value_clean)
        | (ref["_compact_title"] == value_compact)
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def _best_fuzzy_title_match(
    value: str,
    ref: pd.DataFrame,
    title_col: str,
) -> Tuple[pd.Series, float]:

    scores = ref[title_col].apply(lambda title: _combined_similarity(value, title))
    best_idx = scores.idxmax()

    return ref.loc[best_idx], float(scores.loc[best_idx])


def _classify_match(score: float, method: str) -> str:
    if method in {
        "exact_soc_code",
        "exact_title",
        "normalized_exact_title",
    }:
        return "accepted"

    if score >= AUTO_ACCEPT_THRESHOLD:
        return "accepted"

    if score >= REVIEW_THRESHOLD:
        return "review_required"

    return "unmatched"


def _base_unmatched_return(
    method: str,
    score,
    family,
    normalized,
) -> Dict[str, object]:

    return {
        "matched_onet_title": pd.NA,
        "matched_onet_code": pd.NA,
        "onet_match_score": score,
        "onet_match_method": method,
        "onet_match_status": "unmatched",
        "onet_family": family or pd.NA,
        "normalized_title": normalized.canonical_title,
        "title_function": normalized.detected_function,
        "title_level": normalized.detected_level,
        "title_normalization_method": normalized.normalization_method,
    }


def _map_single_role(
    value,
    role_col: str,
    ref: pd.DataFrame,
    title_col: str,
    code_col: Optional[str],
    department_value=None,
) -> Dict[str, object]:

    empty_normalized = normalize_title(value, department_value)

    if pd.isna(value) or str(value).strip() == "":
        return _base_unmatched_return(
            method="missing_role",
            score=pd.NA,
            family=pd.NA,
            normalized=empty_normalized,
        )

    if role_col in {"occupation_code", "matched_onet_code", "soc_code"}:
        code_match = _exact_code_match(str(value), ref, code_col)

        if code_match is not None:
            return {
                "matched_onet_title": code_match[title_col],
                "matched_onet_code": code_match.get(code_col, pd.NA),
                "onet_match_score": 1.0,
                "onet_match_method": "exact_soc_code",
                "onet_match_status": "accepted",
                "onet_family": "code_supplied",
                "normalized_title": empty_normalized.canonical_title,
                "title_function": empty_normalized.detected_function,
                "title_level": empty_normalized.detected_level,
                "title_normalization_method": empty_normalized.normalization_method,
            }

    title_match = _exact_title_match(str(value), ref, title_col)

    if title_match is not None:
        return {
            "matched_onet_title": title_match[title_col],
            "matched_onet_code": title_match.get(code_col, pd.NA),
            "onet_match_score": 1.0,
            "onet_match_method": "exact_title",
            "onet_match_status": "accepted",
            "onet_family": "exact_title",
            "normalized_title": empty_normalized.canonical_title,
            "title_function": empty_normalized.detected_function,
            "title_level": empty_normalized.detected_level,
            "title_normalization_method": empty_normalized.normalization_method,
        }

    normalized = normalize_title(value, department_value)

    role_for_matching = (
        normalized.canonical_title
        if normalized.canonical_title is not None
        else str(value)
    )

    normalized_title_match = _exact_title_match(role_for_matching, ref, title_col)

    if normalized_title_match is not None and normalized.canonical_title is not None:
        return {
            "matched_onet_title": normalized_title_match[title_col],
            "matched_onet_code": normalized_title_match.get(code_col, pd.NA),
            "onet_match_score": 1.0,
            "onet_match_method": "normalized_exact_title",
            "onet_match_status": "accepted",
            "onet_family": normalized.detected_function or pd.NA,
            "normalized_title": normalized.canonical_title,
            "title_function": normalized.detected_function,
            "title_level": normalized.detected_level,
            "title_normalization_method": normalized.normalization_method,
        }

    family = _infer_family(role_for_matching, department_value)
    candidate_ref = _filter_reference_by_family(ref, title_col, family)

    best_match, best_score = _best_fuzzy_title_match(
        role_for_matching,
        candidate_ref,
        title_col,
    )

    method = (
        "normalized_family_constrained_fuzzy_title"
        if normalized.canonical_title is not None and family
        else "family_constrained_fuzzy_title"
        if family
        else "fuzzy_title"
    )

    status = _classify_match(best_score, method)

    if status == "unmatched":
        return _base_unmatched_return(
            method=f"{method}_below_review_threshold",
            score=best_score,
            family=family,
            normalized=normalized,
        )

    return {
        "matched_onet_title": best_match[title_col],
        "matched_onet_code": best_match.get(code_col, pd.NA),
        "onet_match_score": best_score,
        "onet_match_method": method,
        "onet_match_status": status,
        "onet_family": family or pd.NA,
        "normalized_title": normalized.canonical_title,
        "title_function": normalized.detected_function,
        "title_level": normalized.detected_level,
        "title_normalization_method": normalized.normalization_method,
    }


def map_to_onet(
    df: pd.DataFrame,
    onet_reference: pd.DataFrame,
) -> Tuple[pd.DataFrame, OnetMappingReport]:

    out = df.copy()
    role_col = _detect_role_column(out)

    if role_col is None:
        out["matched_onet_title"] = pd.NA
        out["matched_onet_code"] = pd.NA
        out["onet_match_score"] = pd.NA
        out["onet_match_method"] = "no_role_column"
        out["onet_match_status"] = "unmatched"
        out["onet_family"] = pd.NA
        out["normalized_title"] = pd.NA
        out["title_function"] = pd.NA
        out["title_level"] = pd.NA
        out["title_normalization_method"] = pd.NA

        return out, OnetMappingReport(
            role_column=None,
            coverage=0.0,
            accepted_matches=0,
            review_required_matches=0,
            unmatched=len(out),
            exact_matches=0,
            fuzzy_matches=0,
            note="No role/title/code column available. O*NET mapping cannot be performed.",
        )

    ref, title_col, code_col = _prepare_reference(onet_reference)

    department_col = "department" if "department" in out.columns else None

    mapped_rows: List[Dict[str, object]] = []

    for idx, value in out[role_col].items():
        department_value = out.loc[idx, department_col] if department_col else None

        mapped_rows.append(
            _map_single_role(
                value=value,
                role_col=role_col,
                ref=ref,
                title_col=title_col,
                code_col=code_col,
                department_value=department_value,
            )
        )

    mapped_df = pd.DataFrame(mapped_rows, index=out.index)
    out = pd.concat([out, mapped_df], axis=1)

    accepted = int((out["onet_match_status"] == "accepted").sum())
    review_required = int((out["onet_match_status"] == "review_required").sum())
    unmatched = int((out["onet_match_status"] == "unmatched").sum())

    exact = int(
        out["onet_match_method"]
        .isin(["exact_title", "exact_soc_code", "normalized_exact_title"])
        .sum()
    )

    fuzzy = int(
        out["onet_match_method"]
        .astype(str)
        .str.contains("fuzzy", na=False)
        .sum()
    )

    coverage = (
        float(out["onet_match_status"].isin(["accepted", "review_required"]).mean())
        if len(out)
        else 0.0
    )

    return out, OnetMappingReport(
        role_column=role_col,
        coverage=coverage,
        accepted_matches=accepted,
        review_required_matches=review_required,
        unmatched=unmatched,
        exact_matches=exact,
        fuzzy_matches=fuzzy,
        note=(
            "Occupation mapping uses exact code/title matching, title normalization, "
            "family-constrained candidate filtering, fuzzy scoring, and audit status labels. "
            "Accepted matches can be used automatically. Review-required matches should be "
            "inspected before enterprise reporting."
        ),
    )
