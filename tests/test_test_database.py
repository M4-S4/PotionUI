import sqlite3
import threading

import pytest

from tests import conftest


@pytest.mark.parametrize("checkout", ["get_cursor", "get_connection"])
def test_close_waits_for_a_checkout_in_use_and_later_checkouts_fail_clearly(checkout):
    database = conftest.TestDatabase()
    inside = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def hold():
        with getattr(database, checkout)() as handle:
            handle.execute("SELECT 1")
            inside.set()
            release.wait(10)

    def close():
        database.close()
        closed.set()

    holder = threading.Thread(target=hold)
    closer = threading.Thread(target=close)
    holder.start()
    assert inside.wait(10)
    closer.start()

    assert not closed.wait(0.3)
    release.set()
    holder.join(10)
    closer.join(10)

    assert closed.is_set()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        with database.get_cursor():
            pass
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        with database.get_connection():
            pass
