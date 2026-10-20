from src.platform.database.database import db
from src.platform.security.secrets import SecretCipher, get_secret_cipher

SETTING_KEY = "s3_secret_key"


def up():
    with db.get_cursor() as cursor:
        cursor.execute("SELECT id, value FROM settings WHERE key = ?", (SETTING_KEY,))
        row = cursor.fetchone()
        if row is None:
            return
        value = row[1]
        if not value or SecretCipher.is_encrypted(value):
            return
        cursor.execute(
            "UPDATE settings SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (get_secret_cipher().encrypt(value), row[0]),
        )

    print("Migration 061_encrypt_s3_secret_key: applied")


def down():
    print("Migration 061_encrypt_s3_secret_key: nothing to undo")
