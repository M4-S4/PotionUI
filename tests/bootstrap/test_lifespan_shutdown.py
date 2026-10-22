import ast

from tests.bootstrap.lifespan_support import lifespan


def _shutdown_statements():
    body = lifespan().body
    index = next(
        i for i, stmt in enumerate(body) if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield)
    )
    return body[index + 1:]


def _awaited_calls(statements):
    calls = []
    for stmt in statements:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
                calls.append(ast.unparse(node.value))
    return calls


def test_shutdown_stops_model_indexing_and_then_the_digest_pool_before_anything_else_is_torn_down():
    awaited = _awaited_calls(_shutdown_statements())

    assert awaited[:2] == [
        "asyncio.to_thread(container.model_index_manager.indexing.shutdown)",
        "asyncio.to_thread(container.model_index_manager.digests.stop)",
    ]


def test_model_indexing_and_the_digest_pool_are_stopped_before_the_database_pool_is_closed():
    statements = [ast.unparse(stmt) for stmt in _shutdown_statements()]
    stop_indexing = next(i for i, text in enumerate(statements) if "indexing.shutdown" in text)
    stop_digests = next(i for i, text in enumerate(statements) if "digests.stop" in text)
    close = next(i for i, text in enumerate(statements) if "Database.close_pool" in text)

    assert stop_indexing < stop_digests < close


def test_startup_starts_the_digest_pool_before_it_resumes_indexing():
    body = [ast.unparse(stmt) for stmt in lifespan().body]
    start = next(i for i, text in enumerate(body) if "digests.start()" in text)
    resume = next(i for i, text in enumerate(body) if "_resume_model_indexing))" in text)

    assert start < resume
