import ast

from tests.bootstrap.lifespan_support import lifespan


def _startup_calls():
    body = lifespan().body
    index = next(
        i for i, stmt in enumerate(body) if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield)
    )
    return [
        ast.unparse(node.func)
        for stmt in body[:index]
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call)
    ]


def test_startup_fails_chat_turns_left_running_before_the_plugins_are_told_the_app_is_ready():
    calls = _startup_calls()

    assert calls.count("container.turn_store.fail_interrupted") == 1
    assert calls.index("container.turn_store.fail_interrupted") < calls.index("container.plugin_registry.run_ready_hooks")
