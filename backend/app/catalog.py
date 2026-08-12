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
    if "program_source" not in students.columns or students["program_source"].astype(str).eq("Fictional demo enrichment").any():
        students["program"] = "Not available in OULAD"
        students["program_source"] = "Not provided by source dataset"
    enriched["students"] = students
    # Keep only authentic module codes observed in learner history. The former
    # NXT demo catalog is intentionally excluded from the active data model.
    observed = set(map(str, enriched["enrollments"]["course_code"].dropna().unique()))
    courses = enriched["courses"].copy()
    courses = courses[courses["course_code"].astype(str).isin(observed)].copy()
    authentic = load_catalog().query("catalog_source == 'OULAD-derived'").set_index("course_code")
    for column in ["course_name", "department", "level", "credits", "catalog_source"]:
        mapped = courses["course_code"].map(authentic[column])
        courses[column] = mapped.fillna(courses[column] if column in courses else "")
    courses["offered_next_term"] = False
    enriched["courses"] = courses.reset_index(drop=True)
    return enriched


def split_codes(value: str) -> set[str]:
    return {item.strip() for item in str(value).split(";") if item.strip()}
