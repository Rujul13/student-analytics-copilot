from __future__ import annotations

import random

import pandas as pd


MODULES = [
    ("AAA", "Learning Foundations", "Arts & Humanities", 1),
    ("BBB", "Data and Decisions", "Business", 1),
    ("CCC", "Applied Computing", "Computing", 2),
    ("DDD", "Systems Thinking", "Computing", 2),
    ("EEE", "Quantitative Methods", "Mathematics", 2),
    ("FFF", "Research Practice", "Interdisciplinary", 3),
    ("GGG", "Capstone Studio", "Interdisciplinary", 3),
]


def build_development_fixture(student_count: int = 180, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Create a deterministic OULAD-shaped fixture; it is not represented as source data."""
    rng = random.Random(seed)
    students: list[dict] = []
    enrollments: list[dict] = []
    grades: list[dict] = []
    programs = ["Data & Society", "Applied Computing", "Business Analytics"]
    statuses = ["Active", "Active", "Active", "Completed", "Withdrawn"]

    for index in range(student_count):
        student_id = f"S{index + 10001}"
        program = programs[index % len(programs)]
        students.append(
            {
                "student_id": student_id,
                "display_name": f"Student {index + 1:03d}",
                "program": program,
            }
        )
        module_count = rng.randint(2, 5)
        for course_code, _, _, level in rng.sample(MODULES, module_count):
            base = 78 - level * 3 + rng.gauss(0, 12)
            score = max(18, min(100, round(base, 1)))
            status = rng.choice(statuses)
            if status == "Withdrawn":
                score = max(18, min(score, 48))
            result = "Distinction" if score >= 85 else "Pass" if score >= 50 else "Fail"
            if status == "Withdrawn":
                result = "Withdrawn"
            enrollment_id = f"{student_id}-{course_code}-2014J"
            enrollments.append(
                {
                    "enrollment_id": enrollment_id,
                    "student_id": student_id,
                    "course_code": course_code,
                    "presentation": "2014J",
                    "status": status,
                    "final_result": result,
                    "credits": 30 if status == "Completed" or result in {"Pass", "Distinction"} else 0,
                }
            )
            grades.append({"enrollment_id": enrollment_id, "weighted_grade": score})

    courses = pd.DataFrame(
        [
            {
                "course_code": code,
                "course_name": name,
                "department": department,
                "level": level,
                "credits": 30,
                "offered_next_term": True,
            }
            for code, name, department, level in MODULES
        ]
    )
    return {
        "students": pd.DataFrame(students),
        "courses": courses,
        "enrollments": pd.DataFrame(enrollments),
        "grades": pd.DataFrame(grades),
    }

