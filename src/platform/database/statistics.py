import sqlite3

from src.platform.database.database import Database

HOUSEKEEPING_ANALYSIS_LIMIT = 10000


def has_statistics(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_stat1'"
    ).fetchall())


def analyze(conn: sqlite3.Connection, analysis_limit: int = 0) -> None:
    conn.execute(f"PRAGMA analysis_limit = {int(analysis_limit)}").close()
    conn.execute("ANALYZE").close()


def analyze_if_missing(database) -> bool:
    with database.get_connection() as conn:
        if has_statistics(conn):
            return False
        analyze(conn)
        return True


def refresh_statistics(database, analysis_limit: int = HOUSEKEEPING_ANALYSIS_LIMIT) -> None:
    with database.get_connection() as conn:
        analyze(conn, analysis_limit)
    Database.close_pool()
