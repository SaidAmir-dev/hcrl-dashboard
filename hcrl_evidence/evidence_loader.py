from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


EVIDENCE_DIR = Path(__file__).parent


def load_yaml_file(filename: str) -> Dict[str, Any]:
    path = EVIDENCE_DIR / filename

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_hcrl_evidence_library() -> Dict[str, Any]:
    return {
        "mechanisms": load_yaml_file("mechanisms.yaml"),
        "career_progression": load_yaml_file("career_progression.yaml"),
        "compensation": load_yaml_file("compensation.yaml"),
        "interventions": load_yaml_file("interventions.yaml"),
        "references": load_yaml_file("references.yaml"),
    }
