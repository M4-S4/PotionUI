"""Testing a plugin against a scratch database: `scratch_database` makes one, `use_database` makes it the application's database for a `with` block."""

from src.platform.database.testing import scratch_database, use_database

__all__ = ["scratch_database", "use_database"]
