def seed_user(db, user_id="u1", *, username=None, email=None, password_hash="x", account_type="USER"):
    with db.get_cursor() as cursor:
        cursor.execute(
            "INSERT INTO users (id, username, email, password_hash, account_type) VALUES (?, ?, ?, ?, ?)",
            (
                user_id,
                username or user_id,
                email or f"{user_id}@example.test",
                password_hash,
                account_type,
            ),
        )
    return user_id
