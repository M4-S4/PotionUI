import pytest

from src.features.cloud.contracts import CloudBackendConfig, CloudError, CloudHealth, CloudProvider
from src.features.cloud.http import default_error_mapper


class Probing:
    def __init__(self, outcome):
        self.outcome = outcome

    async def probe(self):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class Branded(CloudProvider):
    key = "branded"
    label = "Acme Cloud"
    config_class = CloudBackendConfig
    credits_hint = "An administrator can add credits at acme.test."

    async def discover(self): ...
    async def submit(self, request): ...
    async def poll(self, job): ...


def branded():
    return Branded.__new__(Branded)


async def test_check_reports_the_probe_result():
    health = CloudHealth(ok=True, message="fine")

    assert await CloudProvider.check(Probing(health)) is health


@pytest.mark.parametrize("kind", ["auth", "credits", "refused"])
async def test_check_turns_a_fatal_probe_error_into_an_unhealthy_answer(kind):
    health = await CloudProvider.check(Probing(CloudError(kind, "Plain words.")))

    assert health.ok is False and health.message == "Plain words."


@pytest.mark.parametrize("kind", ["unavailable", "timeout", "rate_limited"])
async def test_check_lets_a_transient_probe_error_through(kind):
    with pytest.raises(CloudError):
        await CloudProvider.check(Probing(CloudError(kind, "Later.")))


async def test_the_default_probe_is_healthy():
    assert (await CloudProvider.probe(Probing(None))).ok is True


def test_the_default_mapper_names_the_brand_it_is_given():
    error = default_error_mapper(401, {}, {}, "Acme")

    assert error.kind == "auth" and error.user_message.startswith("Acme rejected the API key")


def test_the_default_mapper_stays_generic_without_a_brand():
    assert default_error_mapper(401, {}, {}).user_message.startswith("The provider rejected")


def test_the_credits_hint_is_added_only_to_the_credits_message():
    credits = default_error_mapper(402, {"retry-after": "7"}, {}, "Acme", "Add credits at acme.test.")
    other = default_error_mapper(429, {}, {}, "Acme", "Add credits at acme.test.")

    assert credits.user_message == "The Acme account is out of credits. Add credits at acme.test."
    assert credits.retry_after_s == 7.0
    assert "acme.test" not in other.user_message


def test_a_provider_without_its_own_mapping_gets_its_label_and_hint():
    error = branded().map_error(402, {}, {"error": {"message": "empty"}})

    assert error.kind == "credits" and "Acme Cloud" in error.user_message and "acme.test" in error.user_message
    assert "empty" in error.detail


def test_a_branded_refusal_says_the_key_may_lack_access():
    assert "key may not be allowed" in branded().map_error(403, {}, {}).user_message


def test_a_successful_status_maps_to_nothing():
    assert branded().map_error(200, {}, {}) is None


def test_the_user_id_toggle_is_offered_only_to_configs_that_accept_it():
    class Accepting(CloudBackendConfig):
        accepts_user_hash = True

    assert "send_user_hash" not in [spec["name"] for spec in CloudBackendConfig.engine_fields()]
    assert "send_user_hash" in [spec["name"] for spec in Accepting.engine_fields()]
