from tests.platform.database._support import MigrationCase


class TestMigration041ContentSafetyPolicy(MigrationCase):
    MIGRATION = "041_content_safety_policy"
    PREREQUISITES = ("001_baseline", "040_rename_nsfw_setting")

    def _one(self, sql, params=()):
        with self.db.get_cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchone()

    def _setting(self, key):
        return self._one("SELECT value, value_type, type FROM settings WHERE key = ?", (key,))

    def test_seeds_the_policy_settings_with_upgrade_defaults(self):
        self.migration.up()

        assert self._setting("content_policy_nsfw")["value"] == "allowed"
        assert self._setting("content_banned_words")["value"] == "[]"
        assert self._setting("content_banned_words")["value_type"] == "json"
        assert self._setting("content_video_sample_frames")["value"] == "5"
        assert self._setting("content_quarantine_days") is None
        assert self._setting("content_policy_nsfw")["type"] == "SYSTEM"

    def test_legacy_nsfw_row_and_user_overrides_are_removed(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("PRAGMA foreign_keys = OFF")
            cursor.execute("SELECT id FROM settings WHERE key = 'nsfw'")
            setting_id = cursor.fetchone()["id"]
            cursor.execute(
                "INSERT INTO user_settings (id, user_id, setting_id, value) VALUES ('us1', 'u1', ?, 'true')",
                (setting_id,),
            )

        self.migration.up()

        assert self._setting("nsfw") is None
        assert self._one("SELECT 1 FROM user_settings WHERE id = 'us1'") is None

    def test_creates_the_ledger_and_events_tables(self):
        self.migration.up()

        assert self._one("SELECT name FROM sqlite_master WHERE name = 'content_ratings'") is not None
        assert self._one("SELECT name FROM sqlite_master WHERE name = 'content_safety_events'") is not None

    def test_seeds_the_restricted_group_as_a_blocked_system_group(self):
        self.migration.up()

        group = self._one("SELECT * FROM user_groups WHERE id = 'restricted_content'")
        assert group["name"] == "Restricted content"
        assert group["is_system"] == 1
        assert group["content_policy"] == "blocked"

    def test_built_in_groups_have_no_policy(self):
        self.migration.up()

        assert self._one("SELECT content_policy FROM user_groups WHERE id = 'all_users'")["content_policy"] is None

    def test_existing_ratings_are_carried_into_the_ledger(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("PRAGMA foreign_keys = OFF")
            cursor.execute(
                "INSERT INTO files (id, file_path, file_type, user_id) VALUES ('f1', 'g/a.png', 'IMAGE', 'u1')"
            )
            cursor.execute(
                "INSERT INTO files (id, file_path, file_type, user_id) VALUES ('f2', 'g/b.png', 'IMAGE', 'u1')"
            )
            for file_id, tag, score in (
                ("f1", "questionable", 0.4), ("f1", "explicit", 0.4), ("f1", "general", 0.2),
                ("f2", "questionable", 0.1), ("f2", "explicit", 0.1),
            ):
                cursor.execute(
                    """
                    INSERT INTO media_system_tags (id, file_id, tag, category, confidence, provenance)
                    VALUES (?, ?, ?, 'rating', ?, 'wd')
                    """,
                    (f"{file_id}-{tag}", file_id, tag, score),
                )

        self.migration.up()

        flagged = self._one("SELECT * FROM content_ratings WHERE key = 'g/a.png'")
        safe = self._one("SELECT * FROM content_ratings WHERE key = 'g/b.png'")
        assert flagged["state"] == "flagged"
        assert abs(flagged["nsfw_score"] - 0.8) < 1e-9
        assert flagged["source"] == "backfill"
        assert safe["state"] == "safe"

    def test_up_is_idempotent(self):
        self.migration.up()
        self.migration.up()

        assert self._one("SELECT COUNT(*) AS c FROM settings WHERE key = 'content_policy_nsfw'")["c"] == 1
        assert self._one("SELECT COUNT(*) AS c FROM user_groups WHERE id = 'restricted_content'")["c"] == 1
