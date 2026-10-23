import sqlite3
import threading
import weakref
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Tuple

MAX_IDLE_PER_PATH = 8
MAX_PATHS = 4


class PooledConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.open_cursors = weakref.WeakSet()

    def cursor(self, factory=sqlite3.Cursor):
        cursor = super().cursor(factory)
        self.open_cursors.add(cursor)
        return cursor

    def execute(self, sql, parameters=(), /):
        return self.cursor().execute(sql, parameters)

    def executemany(self, sql, parameters, /):
        return self.cursor().executemany(sql, parameters)

    def executescript(self, script, /):
        return self.cursor().executescript(script)

    def close_cursors(self) -> None:
        for cursor in list(self.open_cursors):
            cursor.close()
        self.open_cursors.clear()


class ConnectionLease:
    __slots__ = ("_connection",)

    def __init__(self, connection: Any):
        object.__setattr__(self, "_connection", connection)

    def _live(self) -> Any:
        connection = self._connection
        if connection is None:
            raise sqlite3.ProgrammingError("Cannot operate on a closed database.")
        return connection

    def __getattr__(self, name: str) -> Any:
        return getattr(self._live(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._live(), name, value)

    def __enter__(self) -> "ConnectionLease":
        self._live().__enter__()
        return self

    def __exit__(self, *exc_info) -> Any:
        return self._live().__exit__(*exc_info)

    def revoke(self) -> None:
        object.__setattr__(self, "_connection", None)


_State = Tuple[Any, Any, Any]


def _state(connection: Any) -> _State:
    return connection.isolation_level, connection.row_factory, connection.text_factory


def _restore(connection: Any, state: _State) -> bool:
    try:
        connection.close_cursors()
        if connection.in_transaction:
            connection.rollback()
        isolation_level, row_factory, text_factory = state
        if connection.isolation_level != isolation_level:
            connection.isolation_level = isolation_level
        connection.row_factory = row_factory
        connection.text_factory = text_factory
        return not connection.in_transaction
    except (sqlite3.Error, AttributeError):
        return False


def _close_all(connections: Iterable[Any]) -> None:
    for connection in connections:
        try:
            connection.close()
        except sqlite3.Error:
            pass


class _PathPool:
    __slots__ = ("idle", "members")

    def __init__(self):
        self.idle: List[Any] = []
        self.members: Dict[Any, _State] = {}


class ConnectionPool:
    def __init__(
        self,
        connect: Callable[[str], Any],
        max_idle: int = MAX_IDLE_PER_PATH,
        max_paths: int = MAX_PATHS,
    ):
        self._connect = connect
        self.max_idle = max_idle
        self.max_paths = max_paths
        self._lock = threading.Lock()
        self._paths: "OrderedDict[str, _PathPool]" = OrderedDict()

    def checkout(self, path: str) -> Any:
        retired: List[Any] = []
        with self._lock:
            pool = self._paths.get(path)
            if pool is None:
                pool = self._paths[path] = _PathPool()
                while len(self._paths) > self.max_paths:
                    _, evicted = self._paths.popitem(last=False)
                    retired.extend(evicted.idle)
            else:
                self._paths.move_to_end(path)
            connection = pool.idle.pop() if pool.idle else None
        _close_all(retired)
        if connection is not None:
            return connection
        connection = self._connect(path)
        state = _state(connection)
        with self._lock:
            if self._paths.get(path) is pool:
                pool.members[connection] = state
        return connection

    def checkin(self, path: str, connection: Any) -> None:
        with self._lock:
            pool = self._paths.get(path)
            state = pool.members.get(connection) if pool is not None else None
        if state is not None and _restore(connection, state):
            with self._lock:
                if self._paths.get(path) is pool and len(pool.idle) < self.max_idle:
                    pool.idle.append(connection)
                    return
        if pool is not None:
            with self._lock:
                pool.members.pop(connection, None)
        _close_all([connection])

    def close(self) -> None:
        with self._lock:
            pools = list(self._paths.values())
            self._paths.clear()
        _close_all(connection for pool in pools for connection in pool.idle)
