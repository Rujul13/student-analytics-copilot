from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


CANONICAL_TABLES = ("students", "courses", "enrollments", "grades")


def write_parquet_dataset(frames: dict[str, pd.DataFrame], target: Path) -> None:
    """Write canonical frames as compact, portable Parquet using DuckDB."""
    target.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        for name in CANONICAL_TABLES:
            frame = frames[name]
            connection.register("source_frame", frame)
            path = (target / f"{name}.parquet").as_posix().replace("'", "''")
            connection.execute(f"COPY source_frame TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            connection.unregister("source_frame")
    finally:
        connection.close()


def read_parquet_dataset(source: Path) -> dict[str, pd.DataFrame]:
    """Read canonical Parquet tables through DuckDB, preserving the Pandas agent contract."""
    connection = duckdb.connect()
    try:
        return {
            name: connection.execute(
                "SELECT * FROM read_parquet(?)", [str(source / f"{name}.parquet")]
            ).fetchdf()
            for name in CANONICAL_TABLES
        }
    finally:
        connection.close()


def has_parquet_dataset(source: Path) -> bool:
    return all((source / f"{name}.parquet").exists() for name in CANONICAL_TABLES)


def add_oulad_vle_aggregates(frames: dict[str, pd.DataFrame], source: Path) -> dict[str, pd.DataFrame]:
    """Attach compact historical-engagement aggregates without retaining clickstream rows."""
    path = (source / "studentVle.csv").as_posix().replace("'", "''")
    connection = duckdb.connect()
    connection.execute("SET memory_limit='512MB'")
    connection.execute("SET threads=2")
    try:
        aggregates = connection.execute(f"""
            SELECT
                'OULAD-' || CAST(id_student AS VARCHAR) AS student_id,
                code_module AS course_code,
                code_presentation AS presentation,
                SUM(sum_click)::BIGINT AS vle_total_clicks,
                COUNT(DISTINCT date)::INTEGER AS vle_active_days
            FROM read_csv_auto('{path}', header=true)
            GROUP BY id_student, code_module, code_presentation
        """).fetchdf()
    finally:
        connection.close()
    enriched = {name: frame.copy() for name, frame in frames.items()}
    enriched["enrollments"] = enriched["enrollments"].merge(
        aggregates,
        on=["student_id", "course_code", "presentation"],
        how="left",
    )
    for column in ["vle_total_clicks", "vle_active_days"]:
        enriched["enrollments"][column] = enriched["enrollments"][column].fillna(0).astype(int)
    return enriched


def stage_csv_frames(frames: list[pd.DataFrame]) -> duckdb.DuckDBPyConnection:
    """Create an in-memory analytical staging database for import profiling."""
    connection = duckdb.connect()
    connection.execute("SET memory_limit='256MB'")
    connection.execute("SET threads=2")
    for index, frame in enumerate(frames):
        relation = f"upload_{index}"
        connection.register(f"source_{index}", frame)
        connection.execute(f"CREATE TABLE {relation} AS SELECT * FROM source_{index}")
        connection.unregister(f"source_{index}")
    return connection
