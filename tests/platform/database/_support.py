import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from src.platform.database.database import Database

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[3]
    / "src" / "platform" / "database" / "migrations"
)


def load_migration(stem, database):
    spec = importlib.util.spec_from_file_location(stem, MIGRATIONS_DIR / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    module.db = database
    return module


def _forget_instance():
    Database._instance = None


class MigrationCase(unittest.TestCase):
    MIGRATION = None
    PREREQUISITES = ("001_baseline",)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.addCleanup(_forget_instance)
        self.addCleanup(Database.close_pool)
        self.temp_dir = temp.name
        Database._instance = None
        self.db = Database()
        self.db.db_path = Path(self.temp_dir) / "test.sqlite"
        self.db._initialized = True
        with self.db.get_connection() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS applied_migrations ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "migration_name TEXT UNIQUE NOT NULL, "
                "applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.commit()
        for stem in self.PREREQUISITES:
            load_migration(stem, self.db).up()
        self.pre_state()
        if self.MIGRATION is not None:
            self.migration = load_migration(self.MIGRATION, self.db)

    def pre_state(self):
        pass

    def columns(self, table):
        with self.db.get_cursor() as cursor:
            cursor.execute(f"PRAGMA table_info({table})")
            return {row[1] for row in cursor.fetchall()}
