import pytest

from src.platform.http.targets import is_public_address, origin_of


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.9",
        "192.168.0.1",
        "169.254.169.254",
        "0.0.0.0",
        "100.64.0.1",
        "100.100.100.200",
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",
        "::ffff:100.100.100.200",
        "64:ff9b::7f00:1",
        "64:ff9b::a00:1",
        "2002:7f00:1::",
        "2002:a9fe:a9fe::",
        "224.0.0.1",
        "fe80::1%eth0",
    ],
)
def test_non_public_addresses_are_refused(address):
    assert is_public_address(address) is False


@pytest.mark.parametrize(
    "address",
    ["93.184.216.34", "8.8.8.8", "2606:2800:220:1:248:1893:25c8:1946", "::ffff:8.8.8.8", "64:ff9b::808:808", "2002:808:808::"],
)
def test_public_addresses_pass(address):
    assert is_public_address(address) is True


def test_origin_fills_default_ports_and_lowercases():
    assert origin_of("HTTPS://Example.com/x") == ("https", "example.com", 443)
    assert origin_of("http://example.com:8080/") == ("http", "example.com", 8080)
