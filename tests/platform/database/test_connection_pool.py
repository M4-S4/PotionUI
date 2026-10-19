import contextlib
import sqlite3
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import pytest

from src.features.sessions.version_repository import SessionVersionRepository
from src.platform.database.database import Database
from src.platform.database.pool import MAX_IDLE_PER_PATH, MAX_PATHS, ConnectionPool, PooledConnection


def _sidecars(path):
    return [Path(f"{path}{suffix}") for suffix in ("-wal", "-shm")]


def _fresh_database(monkeypatch, path):
    monkeypatch.setattr(Database, "_instance", None)
    database = Database()
    database.db_path = path
    return database


class _YieldingPaths(OrderedDict):
    def get(self, *args):
        found = super().get(*args)
        time.sleep(0)
        return found


class _TrackingPool(ConnectionPool):
    def __init__(self, connect):
        super().__init__(self._record)
        self._paths = _YieldingPaths()
        self._real_connect = connect
        self.guard = threading.Lock()
        self.opened = []
        self.leased = set()
        self.overlaps = []

    def _record(self, path):
        conn = self._real_connect(path)
        with self.guard:
            self.opened.append(conn)
        return conn

    def checkout(self, path):
        conn = super().checkout(path)
        with self.guard:
            if conn in self.leased:
                self.overlaps.append(conn)
            self.leased.add(conn)
        return conn

    def checkin(self, path, conn):
        with self.guard:
            self.leased.discard(conn)
        super().checkin(path, conn)

    def idle(self, path):
        with self._lock:
            pool = self._paths.get(path)
            return len(pool.idle) if pool is not None else 0


@pytest.fixture
def tracking_pool(monkeypatch):
    pool = _TrackingPool(Database._open)
    monkeypatch.setattr(Database, "_pool", pool)
    yield pool
    pool.close()


@pytest.fixture
def database(tmp_path, monkeypatch):
    database = _fresh_database(monkeypatch, tmp_path / "test.sqlite")
    with database.get_cursor() as cursor:
        cursor.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
    yield database
    Database.close_pool()


def _names(database):
    with database.get_cursor() as cursor:
        cursor.execute("SELECT name FROM items ORDER BY id")
        return [row["name"] for row in cursor.fetchall()]


def _write_elsewhere(database, name):
    conn = sqlite3.connect(database.db_path, timeout=0)
    try:
        conn.execute("INSERT INTO items (name) VALUES (?)", (name,))
        conn.commit()
    finally:
        conn.close()


def test_a_nested_checkout_neither_sees_nor_commits_the_outer_write(database):
    with pytest.raises(RuntimeError):
        with database.get_cursor() as outer:
            outer.execute("INSERT INTO items (name) VALUES ('outer')")
            assert _names(database) == []
            raise RuntimeError("abandon the outer write")

    assert _names(database) == []


def test_an_unfinished_cursor_does_not_pin_the_next_checkout_to_an_old_snapshot(database):
    _write_elsewhere(database, "a")
    _write_elsewhere(database, "b")
    with database.get_connection() as conn:
        unfinished = conn.execute("SELECT name FROM items ORDER BY id")
        assert unfinished.fetchone()["name"] == "a"

    _write_elsewhere(database, "c")

    assert _names(database) == ["a", "b", "c"]
    with pytest.raises(sqlite3.ProgrammingError):
        unfinished.fetchone()


def test_an_exception_rolls_back_and_releases_the_write_lock(database):
    with pytest.raises(RuntimeError):
        with database.get_connection() as conn:
            conn.execute("INSERT INTO items (name) VALUES ('lost')")
            raise RuntimeError("boom")

    _write_elsewhere(database, "other")
    assert _names(database) == ["other"]


def test_an_uncommitted_write_is_discarded_when_the_connection_returns(database):
    with database.get_connection() as conn:
        conn.execute("INSERT INTO items (name) VALUES ('uncommitted')")

    with database.get_connection() as conn:
        assert conn.in_transaction is False
    assert _names(database) == []


def test_the_session_version_repository_leaves_the_connection_transactional(database, monkeypatch):
    monkeypatch.setattr("src.platform.database.database.db", database)
    with database.get_cursor() as cursor:
        cursor.execute(
            "CREATE TABLE session_versions (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
            "version_number INTEGER NOT NULL, payload TEXT NOT NULL, summary TEXT, created_at TIMESTAMP)"
        )

    assert SessionVersionRepository().create_if_changed("s1", {"a": 1}, "first") is not None

    with database.get_connection() as conn:
        assert conn.isolation_level == ""
        conn.execute("INSERT INTO items (name) VALUES ('uncommitted')")
    assert _names(database) == []


def test_a_connection_used_after_it_was_returned_fails_loudly(database):
    with database.get_connection() as conn:
        conn.execute("SELECT 1").fetchall()
    with database.get_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM items")

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        cursor.execute("SELECT 1")


def test_close_pool_releases_the_database_files(database, tmp_path):
    assert _names(database) == []
    assert _sidecars(database.db_path)[1].exists()

    Database.close_pool()

    assert not any(path.exists() for path in _sidecars(database.db_path))
    database.db_path.unlink()
    tmp_path.rmdir()


def test_a_connection_checked_out_during_close_pool_is_closed_on_return(database):
    with database.get_connection() as conn:
        conn.execute("SELECT COUNT(*) FROM items").fetchall()
        Database.close_pool()

    assert not any(path.exists() for path in _sidecars(database.db_path))
    assert _names(database) == []


def test_the_least_recently_used_database_is_released_beyond_the_path_cap(tmp_path, monkeypatch):
    database = _fresh_database(monkeypatch, tmp_path / "unused.sqlite")
    paths = [tmp_path / f"db{index}.sqlite" for index in range(MAX_PATHS + 1)]
    try:
        for path in paths:
            database.db_path = path
            with database.get_cursor() as cursor:
                cursor.execute("CREATE TABLE marker (id INTEGER)")

        assert not any(sidecar.exists() for sidecar in _sidecars(paths[0]))
        assert all(_sidecars(path)[1].exists() for path in paths[1:])
    finally:
        Database.close_pool()


def test_connections_beyond_the_idle_cap_are_closed_on_return(tmp_path):
    opened = []

    def connect(path):
        conn = sqlite3.connect(path, check_same_thread=False, factory=PooledConnection)
        opened.append(conn)
        return conn

    pool = ConnectionPool(connect, max_idle=2)
    path = str(tmp_path / "pool.sqlite")
    held = [pool.checkout(path) for _ in range(3)]
    for conn in held:
        pool.checkin(path, conn)

    assert [pool.checkout(path), pool.checkout(path)] == [held[1], held[0]]
    with pytest.raises(sqlite3.ProgrammingError):
        held[2].execute("SELECT 1")
    assert pool.checkout(path) not in held
    assert len(opened) == 4
    pool.close()
    for conn in opened:
        conn.close()


def test_checkouts_are_never_capped(database):
    with contextlib.ExitStack() as stack:
        leases = [stack.enter_context(database.get_connection()) for _ in range(MAX_IDLE_PER_PATH + 2)]
        leases[0].execute("INSERT INTO items (name) VALUES ('first')")

        assert [lease.execute("SELECT COUNT(*) FROM items").fetchone()[0] for lease in leases] == [1] + [0] * (
            MAX_IDLE_PER_PATH + 1
        )


def test_a_connection_returned_by_one_thread_serves_another(database):
    errors = []

    def write():
        try:
            with database.get_cursor() as cursor:
                cursor.execute("INSERT INTO items (name) VALUES ('from-thread')")
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=write)
    thread.start()
    thread.join(5)

    assert errors == []
    assert _names(database) == ["from-thread"]


def test_factories_changed_on_a_lease_are_restored_for_the_next_checkout(tracking_pool, database):
    with database.get_connection() as conn:
        conn.row_factory = None
        conn.text_factory = bytes

    with database.get_connection() as conn:
        row = conn.execute("SELECT 'text' AS value").fetchone()

    assert row["value"] == "text"
    assert len(tracking_pool.opened) == 1


def test_threads_hammering_one_database_never_share_a_connection(tracking_pool, database):
    threads, iterations = 2 * MAX_IDLE_PER_PATH, 12
    barrier = threading.Barrier(threads)
    errors = []

    def work(index):
        try:
            barrier.wait()
            for iteration in range(iterations):
                name = f"{index}-{iteration}"
                with database.get_cursor() as cursor:
                    cursor.execute("INSERT INTO items (name) VALUES (?)", (name,))
                with database.get_cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM items WHERE name = ?", (name,))
                    assert cursor.fetchone()[0] == 1
                for _ in range(threads):
                    with database.get_connection() as conn:
                        conn.in_transaction
                if iteration % 3 == 0:
                    with database.get_connection() as outer:
                        outer.execute("SELECT COUNT(*) FROM items").fetchall()
                        with database.get_cursor() as inner:
                            inner.execute("SELECT COUNT(*) FROM items")
                if index % 5 == 0 and iteration % 4 == 1:
                    Database.close_pool()
        except Exception as exc:
            errors.append(exc)

    workers = [threading.Thread(target=work, args=(index,)) for index in range(threads)]
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(30)
    finally:
        sys.setswitchinterval(previous_interval)

    assert errors == []
    assert tracking_pool.overlaps == []
    with database.get_cursor() as cursor:
        cursor.execute("SELECT name, COUNT(*) FROM items GROUP BY name")
        counts = dict(cursor.fetchall())
    assert counts == {f"{index}-{iteration}": 1 for index in range(threads) for iteration in range(iterations)}
    assert tracking_pool.idle(str(database.db_path)) <= MAX_IDLE_PER_PATH
