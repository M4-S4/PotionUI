import pytest
from pydantic import BaseModel, ValidationError

from src.platform.security.email_address import EmailAddress


class Holder(BaseModel):
    email: EmailAddress


@pytest.mark.parametrize(
    "value",
    [
        "admin@potionui.local",
        "me@localhost",
        "x@example.test",
        "a@b.internal",
        "user@example.com",
    ],
)
def test_accepts_local_and_reserved_domains(value):
    assert Holder(email=value).email == value


@pytest.mark.parametrize(
    "value",
    ["not-an-email", "a@", "@b.com", "a b@c.d", "a b@localhost", "a@b..local", "a@-x.local", "a@b c", "a@exa mple.local", "a@b_c.local", ""],
)
def test_rejects_malformed(value):
    with pytest.raises(ValidationError) as caught:
        Holder(email=value)
    assert "not a valid email address" in caught.value.errors()[0]["msg"]


def test_normalises_domain_case_and_whitespace_only():
    assert Holder(email="  Me@LocalHost ").email == "Me@localhost"
    assert Holder(email="Ada@Example.COM").email == "Ada@example.com"
