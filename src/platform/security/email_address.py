import re
from typing import Annotated

from email_validator import EmailNotValidError, validate_email
from pydantic import AfterValidator
from pydantic_core import PydanticCustomError

_HOST_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")
_PLACEHOLDER_DOMAIN = "example.com"


def _is_hostname(domain: str) -> bool:
    if not domain or len(domain) > 253:
        return False
    return all(_HOST_LABEL.match(label) for label in domain.split("."))


def _check(value: str) -> str:
    try:
        return validate_email(
            value, check_deliverability=False, globally_deliverable=False
        ).normalized
    except EmailNotValidError:
        local, separator, domain = value.rpartition("@")
        domain = domain.lower()
        if not separator or not _is_hostname(domain):
            raise
        local_part = validate_email(
            f"{local}@{_PLACEHOLDER_DOMAIN}",
            check_deliverability=False,
            globally_deliverable=False,
        ).normalized.rpartition("@")[0]
        return f"{local_part}@{domain}"


def normalize_email_address(value: str) -> str:
    try:
        return _check(value.strip())
    except EmailNotValidError as exc:
        raise PydanticCustomError(
            "value_error",
            "value is not a valid email address: {reason}",
            {"reason": str(exc.args[0])},
        ) from exc


EmailAddress = Annotated[str, AfterValidator(normalize_email_address)]
