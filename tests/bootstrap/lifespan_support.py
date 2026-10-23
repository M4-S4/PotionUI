import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def lifespan():
    tree = ast.parse((ROOT / "src/bootstrap/app.py").read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "lifespan")
