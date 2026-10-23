import threading

from src.features.models.twin_gate import GateResult, SignatureCache, Twin, evaluate


def _twin(signature, location_id=None, sha256=None, path="/x"):
    return Twin(path=path, size=10, location_id=location_id, model_id="m", sha256=sha256, signature=signature)


class TestEvaluate:
    def test_twins_are_split_by_what_their_signature_proves(self):
        same = _twin("s", "l1")
        other = _twin("t", "l2")
        unreadable = _twin(None, "l3")

        result = evaluate("s", [same, other, unreadable])

        assert result == GateResult(equal=(same,), different=(other,), unknown=(unreadable,))

    def test_a_file_whose_own_signature_is_unknown_proves_nothing_about_any_twin(self):
        twins = [_twin("s", "l1"), _twin("t", "l2")]

        result = evaluate(None, twins)

        assert (result.equal, result.different, result.unknown) == ((), (), tuple(twins))

    def test_no_twins_means_nothing_is_contested(self):
        result = evaluate("s", [])

        assert result.contested is False and result.awaited_locations == ()


class TestContested:
    def test_an_equal_signature_contests_the_file(self):
        assert evaluate("s", [_twin("s")]).contested is True

    def test_only_different_signatures_leave_it_alone(self):
        assert evaluate("s", [_twin("t"), _twin("u")]).contested is False

    def test_a_twin_that_cannot_be_read_but_has_a_digest_still_contests_the_file(self):
        assert evaluate("s", [_twin(None, sha256="d" * 64)]).contested is True

    def test_a_twin_that_cannot_be_read_and_has_no_digest_cannot_hold_the_file_back(self):
        assert evaluate("s", [_twin(None)]).contested is False


class TestAwaitedLocations:
    def test_only_equal_twins_without_a_digest_that_are_locations_are_awaited(self):
        result = evaluate(
            "s",
            [
                _twin("s", "pending"),
                _twin("s", "ready", sha256="d" * 64),
                _twin("s", None),
                _twin("t", "different"),
                _twin(None, "unknown"),
            ],
        )

        assert result.awaited_locations == ("pending",)


class TestSignatureCache:
    def test_a_path_is_signed_once_and_the_value_is_shared(self):
        calls = []
        cache = SignatureCache(lambda path, size: calls.append((path, size)) or f"sig:{path}")

        values = [cache.of("/a", 10), cache.of("/a", 10), cache.of("/b", 10)]

        assert values == ["sig:/a", "sig:/a", "sig:/b"]
        assert calls == [("/a", 10), ("/b", 10)]

    def test_an_unreadable_path_is_remembered_as_unknown(self):
        calls = []
        cache = SignatureCache(lambda path, size: calls.append(path))

        assert (cache.of("/a"), cache.of("/a")) == (None, None)
        assert calls == ["/a"]

    def test_threads_asking_for_different_paths_do_not_block_each_other(self):
        started = threading.Event()
        release = threading.Event()

        def compute(path, size):
            if path == "/slow":
                started.set()
                release.wait(5)
            return path

        cache = SignatureCache(compute)
        slow = threading.Thread(target=lambda: cache.of("/slow"), daemon=True)
        slow.start()
        assert started.wait(5)

        assert cache.of("/fast") == "/fast"

        release.set()
        slow.join(5)
