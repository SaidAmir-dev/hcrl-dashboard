"""HCRL Executive Reasoning Engine.

Builds company-specific executive findings from HCRL evidence without
arbitrary weights, hardcoded risk bands, causal claims, firing recommendations,
or fabricated ROI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class ExecutiveReasoningReport:
    domains_analyzed: int
    findings_generated: int
    warnings: List[str]
    errors: List[str]


DOMAIN_LANGUAGE: Mapping[str, Mapping[str, str]] = {
    "Career Progression": {
        "system": "career advancement system",
        "question": "promotion timing, advancement criteria, and internal mobility pathways",
        "action": "promotion governance, career ladders, role progression, and internal mobility pathways",
    },
    "Compensation": {
        "system": "compensation and reward system",
        "question": "pay competitiveness, pay progression, internal equity, and incentives",
        "action": "compensation competitiveness, pay progression, compression, and incentive design",
    },
    "Work Environment": {
        "system": "employee experience and work environment",
        "question": "employee experience, involvement, satisfaction, relationships, and work-life conditions",
        "action": "employee experience, work design, involvement, satisfaction, and work-life conditions",
    },
    "Manager Stability": {
        "system": "management continuity and team leadership",
        "question": "manager continuity, leadership transitions, and team-level management conditions",
        "action": "manager continuity, leadership transitions, and team-level management conditions",
    },
    "Workload": {
        "system": "workload and capacity system",
        "question": "overtime, staffing pressure, scheduling, and workload distribution",
        "action": "workload allocation, overtime concentration, staffing pressure, and scheduling",
    },
    "Travel / Commute Burden": {
        "system": "location, travel, and flexibility model",
        "question": "travel demands, commute burden, location constraints, and flexibility options",
        "action": "travel requirements, location strategy, commute burden, and flexibility options",
    },
    "Training and Development": {
        "system": "learning and workforce development system",
        "question": "training access, development participation, reskilling, and skills pathways",
        "action": "training access, reskilling, development participation, and skills pathways",
    },
    "Occupation": {
        "system": "role and occupation architecture",
        "question": "role-level exposure, labor supply, occupation design, and critical-role protection",
        "action": "role architecture, hiring pipelines, occupation design, and critical-role protection",
    },
    "Department": {
        "system": "department operating model",
        "question": "department operating conditions, leadership practices, and local workforce patterns",
        "action": "department operating conditions, leadership practices, and local workforce patterns",
    },
    "Employee Experience": {
        "system": "employee experience system",
        "question": "onboarding, tenure, employee journey, and experience differences",
        "action": "onboarding, tenure, employee journey, and experience differences",
    },
    "Performance": {
        "system": "performance management system",
        "question": "performance ratings, feedback, rewards, and progression alignment",
        "action": "performance ratings, feedback, rewards, and progression alignment",
    },
    "Education": {
        "system": "workforce education and specialization profile",
        "question": "education mix, specialization, role alignment, and labor-market alternatives",
        "action": "education mix, specialization, role alignment, and skills deployment",
    },
}

DOMAIN_RELATIONSHIPS: Mapping[str, Sequence[str]] = {
    "Career Progression": ("Compensation", "Training and Development", "Performance", "Employee Experience"),
    "Compensation": ("Career Progression", "Performance", "Occupation"),
    "Work Environment": ("Workload", "Manager Stability", "Employee Experience", "Travel / Commute Burden"),
    "Manager Stability": ("Work Environment", "Workload", "Employee Experience"),
    "Workload": ("Work Environment", "Manager Stability", "Travel / Commute Burden"),
    "Training and Development": ("Career Progression", "Occupation", "Education"),
    "Occupation": ("Training and Development", "Compensation", "Education"),
    "Department": ("Manager Stability", "Work Environment", "Workload"),
}


def _first_existing(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    return next((name for name in names if name in df.columns), None)


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _split_variables(value: object) -> List[str]:
    text = _clean(value)
    if not text:
        return []
    parts = [text]
    for sep in ("|", ",", ";"):
        expanded: List[str] = []
        for part in parts:
            expanded.extend(part.split(sep))
        parts = expanded
    seen, output = set(), []
    for part in parts:
        item = _clean(part)
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _join_unique(values: Iterable[object], separator: str = " | ") -> str:
    seen, output = set(), []
    for value in values:
        candidates = value if isinstance(value, (list, tuple, set)) else [value]
        for candidate in candidates:
            item = _clean(candidate)
            if item and item not in seen:
                seen.add(item)
                output.append(item)
    return separator.join(output)


def _currency(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "unavailable" if pd.isna(number) else f"${float(number):,.0f}"


def _normalise_domain(value: object) -> str:
    text = _clean(value)
    aliases = {
        "career progression": "Career Progression",
        "compensation": "Compensation",
        "work environment": "Work Environment",
        "manager stability": "Manager Stability",
        "management quality": "Manager Stability",
        "workload": "Workload",
        "travel / commute burden": "Travel / Commute Burden",
        "travel and commute burden": "Travel / Commute Burden",
        "training and development": "Training and Development",
        "occupation": "Occupation",
        "department": "Department",
        "employee experience": "Employee Experience",
        "performance": "Performance",
        "education": "Education",
    }
    return aliases.get(text.lower(), text or "Unknown")


def _language(domain: str) -> Mapping[str, str]:
    return DOMAIN_LANGUAGE.get(domain, {
        "system": f"{domain.lower()} system",
        "question": f"{domain.lower()} conditions",
        "action": f"{domain.lower()} conditions",
    })


def _normalise_driver_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    domain_col = _first_existing(df, ["driver_group", "workforce_domain", "domain"])
    if domain_col is None:
        return pd.DataFrame()
    out = pd.DataFrame({"driver_group": df[domain_col].map(_normalise_domain)})
    variable_col = _first_existing(df, ["supporting_variables", "matched_company_evidence_variables"])
    out["supporting_variables"] = df[variable_col].map(
        lambda x: _join_unique(_split_variables(x))
    ) if variable_col else ""
    count_col = _first_existing(df, ["evidence_drivers", "company_evidence_link_count"])
    out["evidence_driver_count"] = _numeric(df[count_col]).fillna(0) if count_col else out[
        "supporting_variables"
    ].map(lambda x: len(_split_variables(x)))
    strongest_col = _first_existing(df, ["strongest_association", "absolute_association"])
    out["strongest_association"] = _numeric(df[strongest_col]) if strongest_col else pd.NA
    average_col = _first_existing(df, ["average_association", "mean_association"])
    out["average_association"] = _numeric(df[average_col]) if average_col else pd.NA
    actionability_col = _first_existing(df, ["actionability", "decision_readiness"])
    out["actionability"] = df[actionability_col].map(_clean) if actionability_col else "Unspecified"
    return out


def _normalise_action_table(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    domain_col = _first_existing(df, ["driver_group", "workforce_priority", "workforce_domain"])
    if domain_col is None:
        return pd.DataFrame()
    out = pd.DataFrame({"driver_group": df[domain_col].map(_normalise_domain)})
    exposure_col = _first_existing(df, [
        "linked_modeled_exposure", "exposure_linked_to_intervention_area", "linked_exposure"
    ])
    out["linked_modeled_exposure"] = _numeric(df[exposure_col]).fillna(0.0) if exposure_col else 0.0
    rank_col = _first_existing(df, ["action_rank", "priority_rank", "brief_rank"])
    out["source_priority_rank"] = _numeric(df[rank_col]) if rank_col else pd.NA
    attention_col = _first_existing(df, ["management_attention", "attention_badge"])
    out["management_attention"] = df[attention_col].map(_clean) if attention_col else ""
    return out


def _normalise_prioritization(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    segment_col = _first_existing(df, ["matched_onet_title", "job_title", "segment"])
    if segment_col is None:
        return pd.DataFrame()
    out = pd.DataFrame({"segment": df[segment_col].map(_clean)})
    exposure_col = _first_existing(df, ["total_expected_attrition_cost", "linked_modeled_exposure"])
    out["segment_exposure"] = _numeric(df[exposure_col]).fillna(0.0) if exposure_col else 0.0
    work_type_col = _first_existing(df, ["primary_work_type", "work_type"])
    out["primary_work_type"] = df[work_type_col].map(_clean) if work_type_col else ""
    rank_col = _first_existing(df, ["priority_rank", "economic_rank"])
    out["segment_rank"] = _numeric(df[rank_col]) if rank_col else pd.NA
    return out


def build_domain_evidence_table(
    driver_recommendations_df: pd.DataFrame,
    action_intelligence_df: Optional[pd.DataFrame] = None,
    intervention_economics_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    drivers = _normalise_driver_recommendations(driver_recommendations_df)
    if drivers.empty:
        return pd.DataFrame()

    evidence = drivers.groupby("driver_group", dropna=False).agg(
        supporting_variables=(
            "supporting_variables",
            lambda values: _join_unique(
                variable for value in values for variable in _split_variables(value)
            ),
        ),
        evidence_driver_count=("evidence_driver_count", "max"),
        strongest_association=("strongest_association", "max"),
        average_association=("average_association", "max"),
        actionability=("actionability", lambda values: _join_unique(values)),
    ).reset_index()

    actions = _normalise_action_table(action_intelligence_df)
    if not actions.empty:
        summary = actions.groupby("driver_group", dropna=False).agg(
            linked_modeled_exposure=("linked_modeled_exposure", "max"),
            source_priority_rank=("source_priority_rank", "min"),
            management_attention=("management_attention", lambda values: _join_unique(values)),
        ).reset_index()
        evidence = evidence.merge(summary, on="driver_group", how="left")

    if intervention_economics_df is not None and not intervention_economics_df.empty:
        domain_col = _first_existing(intervention_economics_df, ["driver_group", "workforce_domain"])
        exposure_col = _first_existing(intervention_economics_df, [
            "exposure_linked_to_intervention_area", "linked_modeled_exposure"
        ])
        if domain_col and exposure_col:
            econ = pd.DataFrame({
                "driver_group": intervention_economics_df[domain_col].map(_normalise_domain),
                "economics_exposure": _numeric(intervention_economics_df[exposure_col]).fillna(0.0),
            }).groupby("driver_group", dropna=False)["economics_exposure"].max().reset_index()
            evidence = evidence.merge(econ, on="driver_group", how="left")

    if "linked_modeled_exposure" not in evidence.columns:
        evidence["linked_modeled_exposure"] = 0.0
    if "economics_exposure" in evidence.columns:
        evidence["linked_modeled_exposure"] = evidence[
            ["linked_modeled_exposure", "economics_exposure"]
        ].max(axis=1)
        evidence = evidence.drop(columns=["economics_exposure"])

    evidence["linked_modeled_exposure"] = _numeric(evidence["linked_modeled_exposure"]).fillna(0.0)
    evidence["evidence_driver_count"] = _numeric(evidence["evidence_driver_count"]).fillna(0).astype(int)
    if "source_priority_rank" not in evidence.columns:
        evidence["source_priority_rank"] = pd.NA
    if "management_attention" not in evidence.columns:
        evidence["management_attention"] = ""

    evidence["system_name"] = evidence["driver_group"].map(lambda x: _language(x)["system"])
    evidence["question_focus"] = evidence["driver_group"].map(lambda x: _language(x)["question"])
    evidence["action_focus"] = evidence["driver_group"].map(lambda x: _language(x)["action"])

    def basis(row: pd.Series) -> str:
        facts = [
            bool(_split_variables(row["supporting_variables"])),
            float(row["linked_modeled_exposure"]) > 0,
            not pd.isna(row["strongest_association"]),
        ]
        if all(facts):
            return "multi-source company evidence"
        if any(facts):
            return "partial company evidence"
        return "insufficient company evidence"

    evidence["evidence_basis"] = evidence.apply(basis, axis=1)
    evidence["_has_variables"] = evidence["supporting_variables"].map(
        lambda x: bool(_split_variables(x))
    ).astype(int)
    evidence["_has_exposure"] = (evidence["linked_modeled_exposure"] > 0).astype(int)
    evidence["_has_association"] = evidence["strongest_association"].notna().astype(int)

    evidence = evidence.sort_values(
        ["_has_variables", "_has_exposure", "_has_association", "evidence_driver_count",
         "linked_modeled_exposure", "source_priority_rank", "driver_group"],
        ascending=[False, False, False, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    evidence["evidence_order"] = range(1, len(evidence) + 1)
    return evidence.drop(columns=["_has_variables", "_has_exposure", "_has_association"])


def _top_segment_context(prioritization_df: Optional[pd.DataFrame]) -> Dict[str, object]:
    normalized = _normalise_prioritization(prioritization_df)
    if normalized.empty:
        return {"top_segment": "", "top_segment_exposure": 0.0, "top_segment_work_type": ""}
    top = normalized.sort_values(
        ["segment_exposure", "segment_rank", "segment"],
        ascending=[False, True, True],
        na_position="last",
    ).iloc[0]
    return {
        "top_segment": _clean(top["segment"]),
        "top_segment_exposure": float(top["segment_exposure"]),
        "top_segment_work_type": _clean(top["primary_work_type"]),
    }


def _visible_segments(workforce_df: Optional[pd.DataFrame], max_items: int = 3) -> str:
    if workforce_df is None or workforce_df.empty:
        return ""
    exposure_col = _first_existing(workforce_df, ["expected_attrition_cost", "linked_modeled_exposure"])
    if exposure_col is None:
        return ""
    output: List[str] = []
    for col in ["department", "job_title", "job_role", "matched_onet_title", "job_level", "location", "business_unit", "team"]:
        if col not in workforce_df.columns:
            continue
        summary = workforce_df[workforce_df[col].notna()].groupby(col)[exposure_col].sum().sort_values(ascending=False).head(1)
        if not summary.empty:
            output.append(
                f"{col.replace('_', ' ').title()}: {_clean(summary.index[0])} ({_currency(summary.iloc[0])})"
            )
    return " | ".join(output[:max_items])


def _evidence_summary(row: pd.Series) -> str:
    facts: List[str] = []
    variables = _split_variables(row["supporting_variables"])
    if variables:
        facts.append(f"{len(variables)} observed company variable(s): {', '.join(variables)}")
    if not pd.isna(row["strongest_association"]):
        facts.append(f"a strongest observed association magnitude of {float(row['strongest_association']):.3f}")
    if float(row["linked_modeled_exposure"]) > 0:
        facts.append(f"{_currency(row['linked_modeled_exposure'])} of linked modeled exposure")
    if row["management_attention"]:
        facts.append(f"upstream management attention of {row['management_attention']}")
    return ("; ".join(facts) + ".") if facts else "No usable company-specific evidence was available."


def _related_domains(domain: str, evidence: pd.DataFrame) -> List[str]:
    supported = set(evidence["driver_group"].astype(str))
    return [x for x in DOMAIN_RELATIONSHIPS.get(domain, ()) if x in supported]


def build_executive_findings(
    domain_evidence_df: pd.DataFrame,
    prioritization_df: Optional[pd.DataFrame] = None,
    workforce_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if domain_evidence_df is None or domain_evidence_df.empty:
        return pd.DataFrame()

    evidence = domain_evidence_df.copy()
    segment = _top_segment_context(prioritization_df)
    concentrations = _visible_segments(workforce_df)
    rows: List[Dict[str, object]] = []

    for _, row in evidence.iterrows():
        variables = _split_variables(row["supporting_variables"])
        finding = (
            f"The company's {row['system_name']} is an evidence-supported management review area. "
            f"The available evidence includes {_evidence_summary(row)}"
        )
        if segment["top_segment"]:
            finding += f" The highest-exposure mapped occupation is {segment['top_segment']}"
            if segment["top_segment_exposure"] > 0:
                finding += f", representing {_currency(segment['top_segment_exposure'])} of modeled exposure"
            if segment["top_segment_work_type"]:
                finding += f" and a primary work profile of {segment['top_segment_work_type']}"
            finding += "."
        if concentrations:
            finding += f" Visible concentration points include {concentrations}."

        question = (
            f"Should leadership review {row['question_focus']} within the workforce segments carrying "
            "the strongest observed company evidence and modeled exposure?"
        )
        action = f"Begin a structured review of {row['action_focus']} in the most exposed workforce segments."
        if variables:
            action += " Validate the review against the observed company variables: " + ", ".join(variables) + "."
        if segment["top_segment"]:
            action += f" Start with {segment['top_segment']} before expanding the review company-wide."

        related = _related_domains(row["driver_group"], evidence)
        alternatives = " | ".join(
            f"Review {_language(domain)['action']}" for domain in related
        ) or "No directly related evidence-supported alternative was identified."

        strengths, limits = [], []
        if variables:
            strengths.append(f"supported by {len(variables)} observed company variable(s)")
        else:
            limits.append("no matched company variables were available")
        if float(row["linked_modeled_exposure"]) > 0:
            strengths.append(f"linked to {_currency(row['linked_modeled_exposure'])} of modeled exposure")
        else:
            limits.append("linked modeled exposure was unavailable")
        if not pd.isna(row["strongest_association"]):
            strengths.append("supported by observed statistical association evidence")
        else:
            limits.append("statistical association evidence was unavailable")
        if related:
            limits.append("does not by itself resolve related evidence in " + ", ".join(related))

        tradeoffs = (
            "Evidence strengths: " + ("; ".join(strengths) if strengths else "limited direct evidence") + ". "
            "Trade-offs and limitations: " + ("; ".join(limits) if limits else "none additionally identified") + "."
        )

        rows.append({
            "reasoning_rank": int(row["evidence_order"]),
            "workforce_domain": row["driver_group"],
            "system_under_review": row["system_name"],
            "executive_finding": finding,
            "evidence_summary": _evidence_summary(row),
            "evidence_basis": row["evidence_basis"],
            "executive_question": question,
            "executive_action": action,
            "alternative_actions": alternatives,
            "tradeoffs": tradeoffs,
            "supporting_variables": row["supporting_variables"],
            "evidence_driver_count": int(row["evidence_driver_count"]),
            "strongest_association": row["strongest_association"],
            "average_association": row["average_association"],
            "linked_modeled_exposure": float(row["linked_modeled_exposure"]),
            "source_priority_rank": row["source_priority_rank"],
            "management_attention": row["management_attention"],
            "top_exposed_segment": segment["top_segment"],
            "top_exposed_segment_exposure": segment["top_segment_exposure"],
            "top_exposed_segment_work_type": segment["top_segment_work_type"],
            "causal_status": "Evidence-supported management review; causal effect not established.",
        })

    return pd.DataFrame(rows)


def build_cross_domain_reasoning(findings_df: pd.DataFrame) -> pd.DataFrame:
    if findings_df is None or findings_df.empty:
        return pd.DataFrame()
    available = set(findings_df["workforce_domain"].astype(str))
    by_domain = findings_df.set_index("workforce_domain")
    rows, seen = [], set()
    for primary in findings_df["workforce_domain"].astype(str):
        for related in DOMAIN_RELATIONSHIPS.get(primary, ()):
            if related not in available:
                continue
            pair = tuple(sorted((primary, related)))
            if pair in seen:
                continue
            seen.add(pair)
            first, second = by_domain.loc[primary], by_domain.loc[related]
            rows.append({
                "theme_rank": 0,
                "primary_domain": primary,
                "related_domain": related,
                "combined_theme": f"{_language(primary)['system']} and {_language(related)['system']}",
                "cross_domain_finding": (
                    f"Evidence in {primary} and {related} appears in related workforce systems. "
                    "Leadership should test whether these signals reflect a shared operating mechanism "
                    "before treating either domain in isolation."
                ),
                "combined_linked_exposure": float(first["linked_modeled_exposure"]) + float(second["linked_modeled_exposure"]),
                "supporting_variables": _join_unique([first["supporting_variables"], second["supporting_variables"]]),
                "executive_action": (
                    f"Run a joint review of {_language(primary)['action']} and {_language(related)['action']} "
                    "within the same exposed workforce segments."
                ),
                "causal_status": "Cross-domain relationship is a structured hypothesis, not proof of causality.",
            })
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output = output.sort_values(
        ["combined_linked_exposure", "primary_domain", "related_domain"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    output["theme_rank"] = range(1, len(output) + 1)
    return output


def build_reasoning_brief(
    findings_df: pd.DataFrame,
    cross_domain_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if findings_df is None or findings_df.empty:
        return pd.DataFrame()
    top = findings_df.sort_values("reasoning_rank").iloc[0]
    cross_note = ""
    if cross_domain_df is not None and not cross_domain_df.empty:
        theme = cross_domain_df.sort_values("theme_rank").iloc[0]
        cross_note = f" The leading cross-domain hypothesis connects {theme['primary_domain']} with {theme['related_domain']}."
    return pd.DataFrame([{
        "top_workforce_domain": top["workforce_domain"],
        "top_executive_finding": top["executive_finding"],
        "top_executive_question": top["executive_question"],
        "top_executive_action": top["executive_action"],
        "top_alternative_actions": top["alternative_actions"],
        "top_tradeoffs": top["tradeoffs"],
        "top_evidence_basis": top["evidence_basis"],
        "top_supporting_variables": top["supporting_variables"],
        "top_linked_modeled_exposure": top["linked_modeled_exposure"],
        "executive_summary": (
            f"{top['workforce_domain']} is the first evidence-supported management review area under the current "
            f"lexicographic evidence ordering. {top['executive_action']}{cross_note}"
        ),
        "limitations": (
            "The brief is based on observed company evidence, modeled exposure, and explicit domain relationships. "
            "It does not estimate causal intervention effects, guaranteed ROI, or automatic personnel actions."
        ),
    }])


def build_executive_reasoning_outputs(
    driver_recommendations_df: pd.DataFrame,
    prioritization_df: Optional[pd.DataFrame] = None,
    action_intelligence_df: Optional[pd.DataFrame] = None,
    intervention_economics_df: Optional[pd.DataFrame] = None,
    workforce_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, pd.DataFrame], ExecutiveReasoningReport]:
    warnings: List[str] = []
    errors: List[str] = []
    empty = {
        "domain_evidence": pd.DataFrame(),
        "executive_findings": pd.DataFrame(),
        "cross_domain_reasoning": pd.DataFrame(),
        "executive_reasoning_brief": pd.DataFrame(),
    }
    if driver_recommendations_df is None or driver_recommendations_df.empty:
        errors.append("driver_recommendations_df is required and cannot be empty.")
        return empty, ExecutiveReasoningReport(0, 0, warnings, errors)

    evidence = build_domain_evidence_table(
        driver_recommendations_df,
        action_intelligence_df,
        intervention_economics_df,
    )
    if evidence.empty:
        errors.append("No domain evidence could be built. Check that driver_group is present.")
        return empty, ExecutiveReasoningReport(0, 0, warnings, errors)

    findings = build_executive_findings(evidence, prioritization_df, workforce_df)
    cross = build_cross_domain_reasoning(findings)
    brief = build_reasoning_brief(findings, cross)

    if prioritization_df is None or prioritization_df.empty:
        warnings.append("Prioritization data was unavailable, so occupation-level context was not added.")
    if action_intelligence_df is None or action_intelligence_df.empty:
        warnings.append("Action Intelligence data was unavailable, so linked exposure may be incomplete.")
    if workforce_df is None or workforce_df.empty:
        warnings.append("Workforce-level data was unavailable, so organizational concentration context was not added.")
    warnings.append(
        "Findings are generated from observed company evidence and explicit domain relationships. "
        "They are evidence-supported management reviews, not causal prescriptions."
    )
    warnings.append(
        "Ordering is lexicographic and evidence-first. No weighted score, hardcoded risk band, or arbitrary threshold is used."
    )

    return {
        "domain_evidence": evidence,
        "executive_findings": findings,
        "cross_domain_reasoning": cross,
        "executive_reasoning_brief": brief,
    }, ExecutiveReasoningReport(len(evidence), len(findings), warnings, errors)


def build_executive_reasoning_table(
    driver_recommendations_df: pd.DataFrame,
    prioritization_df: Optional[pd.DataFrame] = None,
    action_intelligence_df: Optional[pd.DataFrame] = None,
    intervention_economics_df: Optional[pd.DataFrame] = None,
    workforce_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, ExecutiveReasoningReport]:
    outputs, report = build_executive_reasoning_outputs(
        driver_recommendations_df,
        prioritization_df,
        action_intelligence_df,
        intervention_economics_df,
        workforce_df,
    )
    return outputs["executive_findings"], report
