import base64
from pathlib import Path
from typing import Any


def encode_media(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def decode_inline(data: Any) -> bytes:
    try:
        return base64.b64decode(data, validate=False)
    except (ValueError, TypeError):
        return b""
