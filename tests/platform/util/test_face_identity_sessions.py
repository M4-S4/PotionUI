import threading
from pathlib import Path

from src.platform.util import face_identity


def test_two_threads_asking_for_the_same_model_build_one_session(monkeypatch):
    monkeypatch.setattr(face_identity, "_sessions", {})
    building = threading.Event()
    release = threading.Event()
    built = []

    def slow_factory(path):
        building.set()
        release.wait(timeout=5)
        built.append(path)
        return object()

    monkeypatch.setattr(face_identity, "_new_session", slow_factory)
    model = Path("face_recognition") / "auraface" / "glintr100.onnx"
    seen = []
    threads = [threading.Thread(target=lambda: seen.append(face_identity._session(model))) for _ in range(2)]
    threads[0].start()
    assert building.wait(timeout=5)
    threads[1].start()
    release.set()
    for thread in threads:
        thread.join(timeout=5)

    assert built == [str(model)]
    assert len(seen) == 2 and seen[0] is seen[1]
    assert face_identity._sessions == {str(model): seen[0]}


def test_a_second_model_gets_its_own_session_without_rebuilding_the_first(monkeypatch):
    monkeypatch.setattr(face_identity, "_sessions", {})
    built = []

    def factory(path):
        built.append(path)
        return object()

    monkeypatch.setattr(face_identity, "_new_session", factory)
    first = face_identity._session(Path("a.onnx"))
    second = face_identity._session(Path("b.onnx"))

    assert face_identity._session(Path("a.onnx")) is first
    assert second is not first
    assert built == ["a.onnx", "b.onnx"]
