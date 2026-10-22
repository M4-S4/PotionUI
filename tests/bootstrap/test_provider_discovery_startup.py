import ast

from tests.bootstrap.lifespan_support import lifespan


def _awaits_ensure(node):
    return any(
        isinstance(n, ast.Await)
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "id", None) == "ensure_providers_discovered"
        for n in ast.walk(node)
    )


def test_startup_awaits_provider_discovery_and_survives_its_failure():
    guarded = [
        n for n in ast.walk(lifespan())
        if isinstance(n, ast.Try) and any(_awaits_ensure(stmt) for stmt in n.body)
    ]

    assert len(guarded) == 1
    assert any(
        h.type is not None and getattr(h.type, "id", None) == "Exception" for h in guarded[0].handlers
    )
    assert not any(isinstance(n, ast.Raise) for h in guarded[0].handlers for n in ast.walk(h))
