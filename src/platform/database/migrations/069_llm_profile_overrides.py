from src.platform.database.database import db

_NAME = "069_llm_profile_overrides"

def up():
    with db.get_cursor() as cursor:
        cursor.execute("PRAGMA table_info(llm_configurations)")
        existing = {row[1] for row in cursor.fetchall()}
        if "profile_overrides" not in existing:
            cursor.execute("ALTER TABLE llm_configurations ADD COLUMN profile_overrides TEXT")

    print(f"Migration {_NAME}: applied")
