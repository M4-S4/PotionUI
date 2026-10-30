import ast
from pathlib import Path

import pytest

from tests.architecture.conftest import is_type_checking_test

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
VENDOR = ROOT / "vendor"
OPS_MODULE = "vendor.gpl.comfyui.ops"
OPS_FILE = VENDOR / "gpl" / "comfyui" / "ops.py"
KERNELS_PACKAGE = Path("platform") / "runtime" / "native" / "kernels"
REDIRECT = "new code must go through src.platform.runtime.native.kernels"

CAPABILITY_PROBE = "get_device_capability"

DIRECT_ATTENTION = "scaled_dot_product_attention"

DIRECT_ATTENTION_ALLOWLIST: dict[str, str] = {
    "src/platform/runtime/native/arch/ltx/duration_head.py":
        "one learned query pooled once per prompt, attention is under 0.1% of the head",
    "src/platform/runtime/native/arch/moge/model.py":
        "camera estimator backbone runs once per input image, not per denoise step",
    "src/platform/runtime/native/text_encoders/_functional.py":
        "text encoders run once per prompt (4% of encoder cost) and pass a scale the dispatcher does not take",
    "src/platform/runtime/native/text_encoders/qwen35.py":
        "text encoder runs once per prompt, 1% of encoder cost",
    "src/platform/runtime/native/vae/ae_2d.py":
        "VAE bottleneck runs once per decode as a single 512-wide head",
    "src/platform/runtime/native/vae/causal_3d.py":
        "VAE bottleneck runs once per decode as a single 384-wide head",
    "src/platform/runtime/native/vae/minimax_h3_audio.py":
        "audio encoder runs once per reference clip in fp32, where the dispatcher always lands on sdpa",
    "src/platform/runtime/native/vae/minimax_h3_video.py":
        "VAE decoder runs once per tile per decode, not per denoise step",
}


def _runtime_nodes(node: ast.AST):
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.If) and is_type_checking_test(child.test):
            for orelse in child.orelse:
                yield orelse
                yield from _runtime_nodes(orelse)
            continue
        yield child
        yield from _runtime_nodes(child)


def _resolve(node: ast.expr, aliases: dict[str, str]) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in aliases:
        return None
    return ".".join([aliases[node.id], *reversed(parts)])


def _ops_references(tree: ast.Module) -> list[int]:
    nodes = list(_runtime_nodes(tree))
    aliases: dict[str, str] = {}
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    head = alias.name.partition(".")[0]
                    aliases[head] = head

    lines: set[int] = set()
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == OPS_MODULE or any(f"{node.module}.{alias.name}" == OPS_MODULE for alias in node.names):
                lines.add(node.lineno)
        elif isinstance(node, ast.Import):
            if any(alias.name == OPS_MODULE for alias in node.names):
                lines.add(node.lineno)
        elif isinstance(node, ast.Constant):
            if node.value == OPS_MODULE:
                lines.add(node.lineno)
        elif isinstance(node, ast.Attribute):
            if _resolve(node, aliases) == OPS_MODULE:
                lines.add(node.lineno)
    return sorted(lines)


def _parsed(base: Path, source_index, skip: Path | None = None):
    for path in source_index.py_files(base):
        if skip is not None and path.is_relative_to(skip):
            continue
        tree = source_index.tree(path)
        assert tree is not None, f"{path} does not parse"
        yield path.relative_to(base.parent).as_posix(), tree


def _ops_problems(base: Path, source_index) -> list[str]:
    problems = []
    for relative, tree in _parsed(base, source_index):
        lines = _ops_references(tree)
        if lines:
            problems.append(f"{relative}:{lines[0]} references {OPS_MODULE}, which is deleted; {REDIRECT}")
    return problems


def _write(base: Path, relative: str, text: str) -> None:
    target = base / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _references(tree: ast.Module, name: str) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            hit = node.attr == name
        elif isinstance(node, ast.Name):
            hit = node.id == name
        elif isinstance(node, ast.Constant):
            hit = node.value == name
        elif isinstance(node, ast.ImportFrom):
            hit = any(alias.name == name for alias in node.names)
        else:
            hit = False
        if hit:
            lines.append(node.lineno)
    return sorted(lines)


def _scan_references(base: Path, source_index, name: str) -> dict[str, int]:
    found: dict[str, int] = {}
    for relative, tree in _parsed(base, source_index, skip=base / KERNELS_PACKAGE):
        lines = _references(tree, name)
        if lines:
            found[relative] = lines[0]
    return found


def _unlisted_entries(found: dict[str, int], allowlist, name: str, suffix: str = "") -> list[str]:
    return [
        f"{path}:{line} references {name}; {REDIRECT}{suffix}"
        for path, line in sorted(found.items())
        if path not in allowlist
    ]


def _stale_entries(found: dict[str, int], allowlist, name: str, label: str) -> list[str]:
    return [
        f"{path} no longer references {name}; remove it from the {label} allowlist"
        for path in sorted(set(allowlist) - found.keys())
    ]


def test_the_vendored_ops_module_stays_deleted():
    assert not OPS_FILE.exists(), f"{OPS_FILE.relative_to(ROOT).as_posix()} is back; {REDIRECT}"


@pytest.mark.parametrize("base", [SOURCE, VENDOR], ids=["src", "vendor"])
def test_nothing_references_the_deleted_vendored_ops_module(base, source_index):
    problems = _ops_problems(base, source_index)
    assert not problems, "references to the deleted vendored ops module:\n" + "\n".join(problems)


@pytest.mark.parametrize("source, lines", [
    ("from vendor.gpl.comfyui.ops import pick_operations\nfrom vendor.gpl.comfyui.ops import cast_to as c\n", [1, 2]),
    ("def later():\n    from vendor.gpl.comfyui.ops import cast_to\n", [2]),
    ("from vendor.gpl.comfyui import ops\nops.pick_operations()\n", [1]),
    ("from vendor.gpl.comfyui import ops as a\na.pick_operations()\n", [1]),
    ("import vendor.gpl.comfyui.ops as a\na.pick_operations()\n", [1]),
    ("import vendor.gpl.comfyui.ops\nvendor.gpl.comfyui.ops.pick_operations()\n", [1, 2]),
    ("from vendor.gpl import comfyui\ncomfyui.ops.pick_operations()\n", [2]),
    ("import vendor.gpl.comfyui as c\nc.ops.pick_operations()\n", [2]),
], ids=[
    "from-import-names", "lazy-from-import", "module-from-import", "module-from-import-aliased",
    "import-aliased", "import-plain", "package-attribute", "package-aliased-attribute",
])
def test_every_spelling_of_the_ops_module_is_recorded_by_line(source, lines):
    assert _ops_references(ast.parse(source)) == lines


def test_string_spellings_of_the_module_count():
    source = (
        "import importlib, sys\n"
        "NAME = 'vendor.gpl.comfyui.ops'\n"
        "importlib.import_module('vendor.gpl.comfyui.ops')\n"
        "sys.modules.get(NAME)\n"
    )

    assert _ops_references(ast.parse(source)) == [2, 3]


def test_type_checking_imports_are_exempt_but_their_else_branch_is_not():
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from vendor.gpl.comfyui.ops import typed_only\n"
        "else:\n"
        "    from vendor.gpl.comfyui.ops import runtime_name\n"
    )

    assert _ops_references(ast.parse(source)) == [5]


def test_sibling_modules_and_relative_imports_do_not_count():
    source = (
        "from vendor.gpl.comfyui.flux.layers import EmbedND\n"
        "from vendor.gpl.comfyui import flux\n"
        "from .ops import local_name\n"
        "from .vendor.gpl.comfyui.ops import sibling_package_name\n"
        "from ...ops.embedding import sinusoidal_embedding\n"
    )

    assert _ops_references(ast.parse(source)) == []


def test_the_ops_scan_covers_the_kernels_package_and_keys_by_repo_relative_path(tmp_path, source_index):
    base = tmp_path / "src"
    _write(base, "features/user.py", "x = 1\nfrom vendor.gpl.comfyui.ops import hidden\n")
    _write(base, "platform/runtime/native/kernels/impl.py", "from vendor.gpl.comfyui.ops import hidden\n")
    _write(base, "platform/runtime/native/other.py", "x = 1\n")

    problems = _ops_problems(base, source_index)

    assert len(problems) == 2
    assert problems[0].startswith("src/features/user.py:2 ")
    assert problems[1].startswith("src/platform/runtime/native/kernels/impl.py:1 ")
    assert all(OPS_MODULE in problem and "src.platform.runtime.native.kernels" in problem for problem in problems)


@pytest.fixture(scope="module")
def capability_references(source_index) -> dict[str, int]:
    return _scan_references(SOURCE, source_index, CAPABILITY_PROBE)


def test_device_capability_is_probed_only_inside_the_kernels_package(capability_references):
    problems = _unlisted_entries(capability_references, frozenset(), CAPABILITY_PROBE)
    assert not problems, "device capability probed outside the kernels package:\n" + "\n".join(problems)


@pytest.mark.parametrize("source, lines", [
    ("import torch\ntorch.cuda.get_device_capability()\n", [2]),
    ("from torch.cuda import get_device_capability\nget_device_capability()\n", [1, 2]),
    ("x = [torch.cuda.get_device_capability(i) for i in range(2)]\n", [1]),
    ("import torch\nprobe = torch.cuda.get_device_capability\n", [2]),
], ids=["attribute-call", "bare-name-call", "comprehension", "bare-reference"])
def test_every_reference_to_the_capability_probe_is_recorded(source, lines):
    assert _references(ast.parse(source), CAPABILITY_PROBE) == lines


@pytest.mark.parametrize("source", [
    "import torch\nprobe = getattr(torch.cuda, 'get_device_capability')\n",
    "from torch.cuda import get_device_capability as probe\n",
], ids=["string-constant", "aliased-import"])
def test_string_and_import_spellings_of_the_capability_probe_are_recorded(source):
    assert len(_references(ast.parse(source), CAPABILITY_PROBE)) == 1


@pytest.mark.parametrize("source", [
    "import torch\ntorch.cuda.get_device_name(0)\n",
    "def get_device_capability_text():\n    return 'torch.cuda.get_device_capability()'\n",
    '"""torch.cuda.get_device_capability() reads the compute capability"""\n',
    "from src.platform.runtime.native.kernels import capabilities\ncapabilities().device(0).capability\n",
], ids=["device-name", "string-and-lookalike-name", "docstring-sentence", "snapshot-read"])
def test_other_device_queries_and_sentences_are_not_capability_references(source):
    assert _references(ast.parse(source), CAPABILITY_PROBE) == []


def test_capability_references_are_listed_by_line_in_order():
    source = "import torch\nx = 1\ntorch.cuda.get_device_capability()\ny = torch.cuda.get_device_capability(1)\n"

    assert _references(ast.parse(source), CAPABILITY_PROBE) == [3, 4]


def test_capability_probe_outside_kernels_names_file_line_and_the_kernels_package(tmp_path, source_index):
    base = tmp_path / "src"
    _write(base, "features/user.py", "import torch\nx = 1\ntorch.cuda.get_device_capability()\ntorch.cuda.get_device_capability(1)\n")
    _write(base, "platform/runtime/native/kernels/impl.py", "import torch\ntorch.cuda.get_device_capability()\n")

    problems = _unlisted_entries(_scan_references(base, source_index, CAPABILITY_PROBE), frozenset(), CAPABILITY_PROBE)

    assert len(problems) == 1
    assert "src/features/user.py:3" in problems[0]
    assert "get_device_capability" in problems[0]
    assert "src.platform.runtime.native.kernels" in problems[0]


@pytest.fixture(scope="module")
def direct_attention(source_index) -> dict[str, int]:
    return _scan_references(SOURCE, source_index, DIRECT_ATTENTION)


def test_no_new_direct_attention_callers_outside_the_kernels_package(direct_attention):
    problems = _unlisted_entries(direct_attention, DIRECT_ATTENTION_ALLOWLIST, DIRECT_ATTENTION, " (attention)")
    assert not problems, "scaled dot product attention called outside the kernels package:\n" + "\n".join(problems)


def test_direct_attention_allowlist_only_shrinks(direct_attention):
    problems = _stale_entries(direct_attention, DIRECT_ATTENTION_ALLOWLIST, DIRECT_ATTENTION, "direct attention")
    assert not problems, "stale direct attention allowlist entries:\n" + "\n".join(problems)


def test_every_direct_attention_entry_gives_a_reason():
    for path, reason in DIRECT_ATTENTION_ALLOWLIST.items():
        assert reason.strip(), path


@pytest.mark.parametrize("source", [
    "from ..kernels import attention\nattention(q, k, v)\n",
    "from .attention import sparse_scaled_dot_product_attention\nsparse_scaled_dot_product_attention(q, k, v)\n",
], ids=["dispatcher-call", "sparse-lookalike"])
def test_dispatcher_calls_and_the_sparse_variant_are_not_direct_attention(source):
    assert _references(ast.parse(source), DIRECT_ATTENTION) == []


def test_unlisted_direct_attention_names_file_line_and_the_kernels_package(tmp_path, source_index):
    base = tmp_path / "src"
    _write(base, "features/user.py", "import torch.nn.functional as F\nx = 1\nF.scaled_dot_product_attention(q, k, v)\n")

    problems = _unlisted_entries(_scan_references(base, source_index, DIRECT_ATTENTION), {}, DIRECT_ATTENTION)

    assert len(problems) == 1
    assert "src/features/user.py:3" in problems[0]
    assert DIRECT_ATTENTION in problems[0]
    assert "src.platform.runtime.native.kernels" in problems[0]


def test_listed_direct_attention_passes_and_an_unused_entry_is_stale(tmp_path, source_index):
    base = tmp_path / "src"
    _write(base, "features/user.py", "import torch.nn.functional as F\nF.scaled_dot_product_attention(q, k, v)\n")
    found = _scan_references(base, source_index, DIRECT_ATTENTION)
    allowlist = {"src/features/user.py": "kept on purpose", "src/features/gone.py": "kept on purpose"}

    assert _unlisted_entries(found, allowlist, DIRECT_ATTENTION) == []
    assert _stale_entries(found, allowlist, DIRECT_ATTENTION, "direct attention") == [
        "src/features/gone.py no longer references scaled_dot_product_attention; "
        "remove it from the direct attention allowlist"
    ]
