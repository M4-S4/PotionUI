#!/usr/bin/env python3
import os
import sys
from getpass import getpass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.platform.security.password import PasswordHasher
from src.features.setup.repository import InstanceClaimRepository
from src.features.users.repository import user_repo

USAGE = (
    "Reset a user's password.\n"
    "\n"
    "Usage:\n"
    "    ./reset_password [username] [newpassword]\n"
    "\n"
    "The username can also come from POTIONUI_RESET_USERNAME. Without one (or with\n"
    "an empty one) the owner account is reset.\n"
    "Without newpassword it is read from POTIONUI_NEW_PASSWORD (checked against\n"
    "POTIONUI_NEW_PASSWORD_REPEAT when that is set), else from a hidden prompt,\n"
    "which keeps it out of shell history and the process table."
)


def _find_user(username: str):
    if username:
        user = user_repo.get_by_username(username)
        if user is None:
            print(f"No user named {username!r}", file=sys.stderr)
        return user
    owner_id = InstanceClaimRepository().owner_user_id()
    user = user_repo.get_by_id(owner_id) if owner_id else None
    if user is None:
        print("This install has no owner account yet", file=sys.stderr)
    return user


def _new_password(args, username: str):
    if len(args) == 2:
        return args[1]
    from_env = os.environ.get("POTIONUI_NEW_PASSWORD")
    if from_env is not None:
        repeat = os.environ.get("POTIONUI_NEW_PASSWORD_REPEAT")
        if repeat is not None and repeat != from_env:
            print("Passwords do not match", file=sys.stderr)
            return None
        return from_env
    password = getpass(f"New password for {username}: ")
    if password != getpass("Repeat: "):
        print("Passwords do not match", file=sys.stderr)
        return None
    return password


def main() -> int:
    args = sys.argv[1:]
    if len(args) > 2:
        print(USAGE, file=sys.stderr)
        return 2

    username = args[0] if args else os.environ.get("POTIONUI_RESET_USERNAME", "")
    user = _find_user(username.strip())
    if user is None:
        return 1

    password = _new_password(args, user.username)
    if password is None:
        return 1
    if not password:
        print("Password must not be empty", file=sys.stderr)
        return 1

    user_repo.update_password(user.id, PasswordHasher().hash(password))
    print(f"Password updated. Sign in as: {user.username}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
