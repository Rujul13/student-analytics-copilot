from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROGRAMS = ("Data & Society", "Applied Computing", "Business Analytics")


def catalog_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "catalog" / "course_catalog.csv"


def load_catalog() -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path(), keep_default_na=False)
    catalog["offered_next_term"] = catalog["offered_next_term"].astype(str).str.lower().eq("true")
    catalog["level"] = pd.to_numeric(catalog["level"], errors="raise").astype(int)
    catalog["credits"] = pd.to_numeric(catalog["credits"], errors="raise").astype(int)
    if catalog["course_code"].duplicated().any():
        raise ValueError("Course catalog contains duplicate course codes")
    return catalog


def enriched_program(student_id: str) -> str:
    digits = re.sub(r"\D", "", student_id)
    stable_number = int(digits or "0")
    return PROGRAMS[stable_number % len(PROGRAMS)]


def apply_catalog_enrichment(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    enriched = {name: frame.copy() for name, frame in frames.items()}
    students = enriched["students"]
    if "program_source" not in students.columns:
        students["program"] = students["student_id"].map(enriched_program)
        students["program_source"] = "Fictional demo enrichment"
    enriched["students"] = students
    enriched["courses"] = load_catalog()
    return enriched


def split_codes(value: str) -> set[str]:
    return {item.strip() for item in str(value).split(";") if item.strip()}

