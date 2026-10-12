import time
from datetime import timedelta
from typing import Optional

from src.platform.security.token import TokenCodec

_SPENT_CLAIMS = ("exp", "iat")


def renewed_token(codec: TokenCodec, token: str, now: Optional[float] = None) -> Optional[str]:
    claims = codec.decode_claims(token)
    if claims is None:
        return None
    issued_at = claims.get("iat")
    expires_at = claims.get("exp")
    if not isinstance(issued_at, (int, float)) or not isinstance(expires_at, (int, float)):
        return None
    lifetime = expires_at - issued_at
    if lifetime <= 0:
        return None
    remaining = expires_at - (time.time() if now is None else now)
    if remaining >= lifetime / 2:
        return None
    carried = {key: value for key, value in claims.items() if key not in _SPENT_CLAIMS}
    return codec.create_access_token(carried, timedelta(seconds=lifetime))
