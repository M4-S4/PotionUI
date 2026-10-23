"""
Integration test for test fixtures.

This test file verifies that all fixtures work correctly together
and provides examples of how to use them in tests.
"""

import pytest
from pathlib import Path


class TestStorageFixtures:
    """Test storage-related fixtures"""

    def test_file_service_can_save_files(self, file_service, fake_image_bytes):
        """Verify file_service can save files in test storage"""
        generation_id = "test_gen_123"
        file_path, metadata = file_service.save_file(
            generation_id=generation_id,
            file_data=fake_image_bytes,
            extension="png",
            prefix="0"
        )

        assert file_path is not None
        assert metadata is not None
        assert Path(file_path).exists()
        assert metadata['file_type'] == 'IMAGE'


class TestFixtureIsolation:
    """Test that fixtures provide proper isolation between tests"""

    def test_database_isolation_first_test(self, test_db):
        """First test - insert data and verify"""
        with test_db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO users (id, username, email, password_hash)
                VALUES ('test_id_1', 'isolation_test_1', 'test1@example.com', 'hash')
            """)
            cursor.execute("SELECT COUNT(*) as count FROM users WHERE username LIKE 'isolation_test%'")
            result = cursor.fetchone()
            assert result['count'] == 1

    def test_database_isolation_second_test(self, test_db):
        """Second test - verify previous test data doesn't exist"""
        with test_db.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM users WHERE username LIKE 'isolation_test%'")
            result = cursor.fetchone()
            # Should be 0 because each test gets a fresh database
            assert result['count'] == 0

    def test_storage_isolation_first_test(self, test_storage, fake_image_bytes):
        """First test - create file in storage"""
        test_file = test_storage / "tmp" / "test_file_1.png"
        test_file.write_bytes(fake_image_bytes)
        assert test_file.exists()

    def test_storage_isolation_second_test(self, test_storage):
        """Second test - verify previous test file doesn't exist"""
        test_file = test_storage / "tmp" / "test_file_1.png"
        # Should not exist because each test gets fresh storage
        assert not test_file.exists()
