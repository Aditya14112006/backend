"""
SQLite connection handling for the Ocean Intelligence backend.

Kept intentionally minimal at the foundation stage: just enough to prove
connectivity via /health and to hold the tables that later steps (dataset
upload, route logging) will read/write.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "ocean_intelligence.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            dataset_type TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            row_count INTEGER,
            columns TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS route_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_name TEXT NOT NULL,
            destination_name TEXT NOT NULL,
            queried_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()
