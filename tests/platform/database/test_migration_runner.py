import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the project root to Python path. This file is three levels below the
# root — a shorter relative path lands on tests/, whose plugins/__init__.py
# then shadows the real plugins/ namespace for every later-collected test.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.platform.database.database import Database
from src.platform.database.migration_runner import MigrationRunner


class TestMigrationRunner(unittest.TestCase):
    
    def setUp(self):
        """Set up test migration manager with isolated database"""
        # Create temporary database for testing
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db_path = Path(self.temp_dir) / "test.sqlite"
        
        # Create isolated database instance
        self.db = self._create_test_database(self.temp_db_path)
        
        # Create a temporary migrations directory for testing
        self.temp_migrations_dir = tempfile.mkdtemp()
        self.migrations_dir = Path(self.temp_migrations_dir)
        
        # Create test migration manager with isolated database
        self.migration_runner = MigrationRunner()
        self.migration_runner.migrations_dir = self.migrations_dir
        
        # Override the migration manager's database methods to use test database
        self._patch_migration_runner_db()
    
    def tearDown(self):
        """Clean up temporary files"""
        # Clean up database
        Database.close_pool()
        if self.temp_db_path.exists():
            self.temp_db_path.unlink()
        os.rmdir(self.temp_dir)
        
        # Reset database singleton
        Database._instance = None
        
        # Clean up temporary migrations directory
        import shutil
        if self.migrations_dir.exists():
            shutil.rmtree(self.migrations_dir)
    
    def _create_test_database(self, db_path: Path) -> Database:
        """Create a database instance for testing"""
        # Reset singleton instance to ensure fresh database for each test
        Database._instance = None
        
        # Create new database instance
        db = Database()
        db.db_path = db_path
        db.db_path.parent.mkdir(exist_ok=True)
        db._initialized = True  # Mark as initialized to avoid conflicts
        
        # Monkey patch the global db instance used by migration manager
        import importlib
        for module_name in ("src.platform.database.database", "src.platform.database.migration_runner"):
            module = importlib.import_module(module_name)
            self.addCleanup(setattr, module, "db", module.db)
            module.db = db

        return db
    
    def _patch_migration_runner_db(self):
        """Patch the migration manager to use the test database"""
        # Store original methods
        self.original_get_applied_migrations = self.migration_runner.get_applied_migrations
        self.original_mark_migration_applied = self.migration_runner._mark_migration_applied
        
        # Override methods to use test database
        def get_applied_migrations():
            with self.db.get_cursor() as cursor:
                # Create migrations table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS applied_migrations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        migration_name TEXT UNIQUE NOT NULL,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("SELECT migration_name FROM applied_migrations ORDER BY migration_name")
                return [row[0] for row in cursor.fetchall()]
        
        def mark_migration_applied(migration_name: str):
            with self.db.get_cursor() as cursor:
                cursor.execute("INSERT INTO applied_migrations (migration_name) VALUES (?)", (migration_name,))
        
        # Apply patches
        self.migration_runner.get_applied_migrations = get_applied_migrations
        self.migration_runner._mark_migration_applied = mark_migration_applied
    
    def _create_test_migration(self, name: str, up_function: str = None):
        """Create a test migration file"""
        if up_function is None:
            up_function = """
def up():
    from src.platform.database.database import db
    with db.get_cursor() as cursor:
        cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY)")
"""
        
        migration_content = f"""# Test migration: {name}
{up_function}
"""
        
        migration_file = self.migrations_dir / f"{name}.py"
        migration_file.write_text(migration_content)
        return migration_file
    
    def test_get_applied_migrations_empty(self):
        """Test getting applied migrations when none exist"""
        applied = self.migration_runner.get_applied_migrations()
        self.assertEqual(applied, [])
        
        # Verify migrations table was created
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='applied_migrations'")
            result = cursor.fetchone()
            self.assertIsNotNone(result)
    
    def test_get_applied_migrations_with_data(self):
        """Test getting applied migrations when some exist"""
        # Call method first to create table
        self.migration_runner.get_applied_migrations()
        
        # Add some test migration records
        with self.db.get_cursor() as cursor:
            cursor.execute("INSERT INTO applied_migrations (migration_name) VALUES (?)", ("001_first_migration",))
            cursor.execute("INSERT INTO applied_migrations (migration_name) VALUES (?)", ("002_second_migration",))
        
        applied = self.migration_runner.get_applied_migrations()
        self.assertEqual(applied, ["001_first_migration", "002_second_migration"])
    
    def test_get_available_migrations(self):
        """Test getting available migration files"""
        # Create test migration files
        self._create_test_migration("001_first_migration")
        self._create_test_migration("002_second_migration")
        self._create_test_migration("003_third_migration")
        
        # Create __init__.py (should be ignored)
        (self.migrations_dir / "__init__.py").write_text("")
        
        available = self.migration_runner.get_available_migrations()
        expected = ["001_first_migration", "002_second_migration", "003_third_migration"]
        self.assertEqual(available, expected)
    
    def test_get_available_migrations_empty(self):
        """Test getting available migrations when none exist"""
        available = self.migration_runner.get_available_migrations()
        self.assertEqual(available, [])
    
    def test_run_migrations_no_pending(self):
        """Test running migrations when none are pending"""
        # Create and mark migrations as applied
        self._create_test_migration("001_test_migration")
        
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS applied_migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration_name TEXT UNIQUE NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("INSERT INTO applied_migrations (migration_name) VALUES (?)", ("001_test_migration",))
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            self.migration_runner.run_migrations()
            # run_migrations returns silently when nothing is pending (it no
            # longer prints "No pending migrations"), so no migration ran.
            for call in mock_print.call_args_list:
                self.assertNotIn("Running", str(call))
                self.assertNotIn("Applying migration", str(call))
    
    def test_run_migrations_with_pending(self):
        """Test running pending migrations"""
        # Create test migrations
        self._create_test_migration("001_first_migration", """
def up():
    from src.platform.database.database import db
    with db.get_cursor() as cursor:
        cursor.execute("CREATE TABLE first_table (id INTEGER PRIMARY KEY)")
""")
        
        self._create_test_migration("002_second_migration", """
def up():
    from src.platform.database.database import db
    with db.get_cursor() as cursor:
        cursor.execute("CREATE TABLE second_table (id INTEGER PRIMARY KEY)")
""")
        
        # Mark first migration as applied
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS applied_migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration_name TEXT UNIQUE NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("INSERT INTO applied_migrations (migration_name) VALUES (?)", ("001_first_migration",))
        
        # Run migrations
        with patch('builtins.print') as mock_print:
            self.migration_runner.run_migrations()
        
        # Verify second migration was applied
        applied = self.migration_runner.get_applied_migrations()
        self.assertIn("002_second_migration", applied)
        
        # Verify table was created
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='second_table'")
            result = cursor.fetchone()
            self.assertIsNotNone(result)
    
    def test_a_run_that_creates_an_index_collects_planner_statistics(self):
        self._create_test_migration("001_indexed_table", """
def up():
    from src.platform.database.database import db
    with db.get_cursor() as cursor:
        cursor.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, kind TEXT)")
        cursor.executemany("INSERT INTO items (kind) VALUES (?)", [("a",), ("a",), ("b",), ("b",)])
        cursor.execute("CREATE INDEX idx_items_kind ON items (kind)")
""")

        with patch('builtins.print'):
            self.migration_runner.run_migrations()

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT stat FROM sqlite_stat1 WHERE idx = 'idx_items_kind'")
            self.assertEqual(cursor.fetchone()[0], "4 2")

    def test_a_run_that_redefines_an_index_under_its_old_name_collects_planner_statistics(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, kind TEXT)")
            cursor.executemany("INSERT INTO items (kind) VALUES (?)", [("a",), ("a",), ("b",), ("b",)])
            cursor.execute("CREATE INDEX idx_items_kind ON items (kind)")
        self._create_test_migration("001_widen_items_kind", """
def up():
    from src.platform.database.database import db
    with db.get_cursor() as cursor:
        cursor.execute("DROP INDEX idx_items_kind")
        cursor.execute("CREATE INDEX idx_items_kind ON items (kind, id)")
""")

        with patch('builtins.print'):
            self.migration_runner.run_migrations()

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT stat FROM sqlite_stat1 WHERE idx = 'idx_items_kind'")
            self.assertEqual(cursor.fetchone()[0], "4 2 1")

    def test_a_run_that_collects_planner_statistics_recycles_pooled_connections(self):
        self._create_test_migration("001_indexed_table", """
def up():
    from src.platform.database.database import db
    with db.get_cursor() as cursor:
        cursor.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, kind TEXT)")
        cursor.execute("CREATE INDEX idx_items_kind ON items (kind)")
""")

        with patch('builtins.print'), patch.object(Database, "close_pool") as close_pool:
            self.migration_runner.run_migrations()

        close_pool.assert_called_once_with()

    def test_a_run_that_only_changes_rows_collects_no_planner_statistics(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, kind TEXT)")
            cursor.execute("CREATE INDEX idx_items_kind ON items (kind)")
        self._create_test_migration("001_seed_items", """
def up():
    from src.platform.database.database import db
    with db.get_cursor() as cursor:
        cursor.execute("INSERT INTO items (kind) VALUES ('a')")
""")

        with patch('builtins.print'):
            self.migration_runner.run_migrations()

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE name = 'sqlite_stat1'")
            self.assertIsNone(cursor.fetchone())

    def test_run_migration_without_up_function(self):
        """Test running migration that doesn't have up function"""
        # Create migration without up function
        self._create_test_migration("001_bad_migration", "# No up function here")
        
        with self.assertRaises(AttributeError) as context:
            self.migration_runner._run_migration("001_bad_migration")
        
        self.assertIn("must have an 'up' function", str(context.exception))
    
    def test_run_migration_with_syntax_error(self):
        """Test running migration with syntax error"""
        # Create migration with syntax error
        migration_content = """
def up():
    this is not valid python syntax !@#$
"""
        migration_file = self.migrations_dir / "001_bad_syntax.py"
        migration_file.write_text(migration_content)
        
        with self.assertRaises(SyntaxError):
            self.migration_runner._run_migration("001_bad_syntax")
    
    def test_mark_migration_applied(self):
        """Test marking migration as applied"""
        migration_name = "001_test_migration"
        
        # Create table first
        self.migration_runner.get_applied_migrations()
        
        # Mark migration as applied
        self.migration_runner._mark_migration_applied(migration_name)
        
        # Verify it was marked as applied
        applied = self.migration_runner.get_applied_migrations()
        self.assertIn(migration_name, applied)
    
    def test_mark_migration_applied_duplicate(self):
        """Test marking migration as applied when already applied"""
        migration_name = "001_test_migration"
        
        # Create table first
        self.migration_runner.get_applied_migrations()
        
        # Mark migration as applied twice
        self.migration_runner._mark_migration_applied(migration_name)
        
        with self.assertRaises(Exception):  # Should raise unique constraint error
            self.migration_runner._mark_migration_applied(migration_name)
    
    def test_integration_full_migration_cycle(self):
        """Test complete migration cycle"""
        # Create test migrations
        self._create_test_migration("001_create_users", """
def up():
    from src.platform.database.database import db
    with db.get_cursor() as cursor:
        cursor.execute('''
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL
            )
        ''')
""")
        
        self._create_test_migration("002_add_timestamps", """
def up():
    from src.platform.database.database import db
    with db.get_cursor() as cursor:
        cursor.execute('''
            ALTER TABLE users ADD COLUMN created_at TEXT DEFAULT (datetime('now'))
        ''')
        cursor.execute('''
            ALTER TABLE users ADD COLUMN updated_at TEXT DEFAULT (datetime('now'))
        ''')
""")
        
        # Initially no migrations applied
        self.assertEqual(len(self.migration_runner.get_applied_migrations()), 0)
        
        # Run migrations
        self.migration_runner.run_migrations()
        
        # Verify both migrations were applied
        applied = self.migration_runner.get_applied_migrations()
        self.assertEqual(len(applied), 2)
        self.assertIn("001_create_users", applied)
        self.assertIn("002_add_timestamps", applied)
        
        # Verify tables were created correctly
        with self.db.get_cursor() as cursor:
            # Check users table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check table structure
            cursor.execute("PRAGMA table_info(users)")
            columns = [row[1] for row in cursor.fetchall()]
            expected_columns = ['id', 'username', 'email', 'created_at', 'updated_at']
            for col in expected_columns:
                self.assertIn(col, columns)
        
        # Run migrations again - should be no-op
        with patch('builtins.print') as mock_print:
            self.migration_runner.run_migrations()
            # run_migrations returns silently when nothing is pending (it no
            # longer prints "No pending migrations"), so no migration ran.
            for call in mock_print.call_args_list:
                self.assertNotIn("Running", str(call))
                self.assertNotIn("Applying migration", str(call))


class TestEncryptS3SecretKeyThroughRunner(unittest.TestCase):

    def setUp(self):
        import importlib
        import shutil
        from src.platform.security.secrets import SecretCipher, configure_secret_cipher, generate_key

        self.temp_dir = Path(tempfile.mkdtemp())
        Database._instance = None
        self.db = Database()
        self.db.db_path = self.temp_dir / "test.sqlite"
        self.db._initialized = True
        for module_name in ("src.platform.database.database", "src.platform.database.migration_runner"):
            module = importlib.import_module(module_name)
            self.addCleanup(setattr, module, "db", module.db)
            module.db = self.db
        self.cipher = SecretCipher([generate_key()])
        configure_secret_cipher(self.cipher)
        self.addCleanup(configure_secret_cipher, None)
        self.addCleanup(shutil.rmtree, self.temp_dir, True)
        self.addCleanup(setattr, Database, "_instance", None)
        self.addCleanup(Database.close_pool)

        source = Path(__file__).resolve().parents[3] / "src" / "platform" / "database" / "migrations"
        self.source = source
        self.migrations_dir = self.temp_dir / "migrations"
        self.migrations_dir.mkdir()
        shutil.copy(source / "001_baseline.py", self.migrations_dir / "001_baseline.py")
        self.runner = MigrationRunner()
        self.runner.migrations_dir = self.migrations_dir

    def test_pending_migration_encrypts_a_plaintext_secret(self):
        import shutil

        with patch("builtins.print"):
            self.runner.run_migrations()
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE settings SET value = 'plain-secret' WHERE key = 's3_secret_key'")

        shutil.copy(
            self.source / "061_encrypt_s3_secret_key.py",
            self.migrations_dir / "061_encrypt_s3_secret_key.py",
        )
        with patch("builtins.print"):
            self.runner.run_migrations()

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT value FROM settings WHERE key = 's3_secret_key'")
            stored = cursor.fetchone()[0]
        self.assertNotEqual(stored, "plain-secret")
        self.assertEqual(self.cipher.decrypt(stored, context="test"), "plain-secret")
        self.assertIn("061_encrypt_s3_secret_key", self.runner.get_applied_migrations())


if __name__ == '__main__':
    unittest.main()
