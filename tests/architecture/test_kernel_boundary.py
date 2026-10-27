import ast
from pathlib import Path

import pytest

from tests.architecture.conftest import is_type_checking_test

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
OPS_MODULE = "vendor.gpl.comfyui.ops"
WHOLE_MODULE = "<module>"
KERNELS_PACKAGE = Path("platform") / "runtime" / "native" / "kernels"
REDIRECT = "new code must go through src.platform.runtime.native.kernels"

ALLOWLIST = frozenset({
    ("src/features/llm/native_te_adoption.py", "_parse_comfy_quant"),
    ("src/features/llm/native_te_adoption.py", "cast_to"),
    ("src/features/llm/native_te_adoption.py", "dequantize_nvfp4"),
    ("src/pipelines/pipes/_shared/generation/dit_placement.py", "_fp8_matmul_enabled"),
    ("src/pipelines/pipes/_shared/generation/dit_placement.py", "_lora_fused_enabled"),
    ("src/pipelines/pipes/_shared/generation/dit_placement.py", "_nvfp4_matmul_enabled"),
    ("src/pipelines/pipes/_shared/generation/dit_placement.py", "partition_output_branch_deltas"),
    ("src/pipelines/pipes/model_loader/minimax_h3/vdn.py", "pick_operations"),
    ("src/pipelines/pipes/model_loader/minimax_music3/te_loader.py", "detect_quant_format"),
    ("src/pipelines/pipes/model_loader/minimax_music3/te_loader.py", "pick_operations"),
    ("src/platform/runtime/native/__init__.py", "detect_quant_format"),
    ("src/platform/runtime/native/__init__.py", "pick_operations"),
    ("src/platform/runtime/native/arch/trellis2/load.py", "manual_cast"),
    ("src/platform/runtime/native/engine.py", "QUANT_FP8_SCALED"),
    ("src/platform/runtime/native/engine.py", "detect_quant_format"),
    ("src/platform/runtime/native/engine.py", "disable_weight_init"),
    ("src/platform/runtime/native/engine.py", "fp8_ops"),
    ("src/platform/runtime/native/engine.py", "log_lora_path_summary"),
    ("src/platform/runtime/native/engine.py", "manual_cast"),
    ("src/platform/runtime/native/engine.py", "pick_operations"),
    ("src/platform/runtime/native/io/safetensors_loader.py", "_extract_convrot_config"),
    ("src/platform/runtime/native/io/safetensors_loader.py", "_parse_comfy_quant"),
    ("src/platform/runtime/native/io/state_dict_utils.py", "_parse_comfy_quant"),
    ("src/platform/runtime/native/ops/model_tools/formats.py", "_build_convrot_hadamard"),
    ("src/platform/runtime/native/ops/model_tools/formats.py", "_convrot_unrotate_weight"),
    ("src/platform/runtime/native/ops/model_tools/formats.py", "_quantize_nvfp4_dynamic"),
    ("src/platform/runtime/native/ops/model_tools/formats.py", "dequantize_nvfp4"),
    ("src/platform/runtime/native/ops/model_tools/skeleton.py", "pick_operations"),
    ("src/platform/runtime/native/optimizations/compile.py", "CastWeightBiasOp"),
    ("src/platform/runtime/native/optimizations/compile.py", "Fp8ScaledLinear"),
    ("src/platform/runtime/native/optimizations/compile.py", "QUANT_FP8_SCALED"),
    ("src/platform/runtime/native/optimizations/compile.py", "_FP8_DTYPES"),
    ("src/platform/runtime/native/optimizations/compile.py", "_fp8_matmul_enabled"),
    ("src/platform/runtime/native/text_encoders/loader.py", "detect_quant_format"),
    ("src/platform/runtime/native/text_encoders/loader.py", "pick_operations"),
    ("src/platform/settings/runtime_flags.py", "<module>"),
})

CAPABILITY_PROBE = "get_device_capability"

CAPABILITY_ALLOWLIST: frozenset[str] = frozenset()

DIRECT_ATTENTION = "scaled_dot_product_attention"

DIRECT_ATTENTION_ALLOWLIST: dict[str, str] = {
    "src/platform/runtime/native/arch/ltx/duration_head.py":
        "one learned query pooled once per prompt, attention is under 0.1% of the head",
    "src/platform/runtime/native/arch/minimax_h3/model.py":
        "query-chunked rescue after an accelerated kernel ran out of memory must be plain sdpa",
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


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


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


def _usages(tree: ast.Module) -> list[tuple[str, int]]:
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

    found: list[tuple[str, int]] = []
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                if node.module == OPS_MODULE:
                    found.append((alias.name, node.lineno))
                elif f"{node.module}.{alias.name}" == OPS_MODULE:
                    found.append((WHOLE_MODULE, node.lineno))
        elif isinstance(node, ast.Import):
            if any(alias.name == OPS_MODULE for alias in node.names):
                found.append((WHOLE_MODULE, node.lineno))
        elif isinstance(node, ast.Constant):
            if node.value == OPS_MODULE:
                found.append((WHOLE_MODULE, node.lineno))
        elif isinstance(node, ast.Attribute):
            if _resolve(node, aliases) == OPS_MODULE:
                found.append((WHOLE_MODULE, node.lineno))
            if _is_private(node.attr) and _resolve(node.value, aliases) == OPS_MODULE:
                found.append((node.attr, node.lineno))
    return sorted(found, key=lambda usage: (usage[1], usage[0]))


def _walk_outside_kernels(base: Path, source_index):
    exempt = base / KERNELS_PACKAGE
    for path in source_index.py_files(base):
        if path.is_relative_to(exempt):
            continue
        tree = source_index.tree(path)
        assert tree is not None, f"{path} does not parse"
        yield path.relative_to(base.parent).as_posix(), tree


def _scan(base: Path, source_index) -> dict[tuple[str, str], int]:
    found: dict[tuple[str, str], int] = {}
    for relative, tree in _walk_outside_kernels(base, source_index):
        for name, line in _usages(tree):
            found.setdefault((relative, name), line)
    return found


def _unlisted(found: dict[tuple[str, str], int], allowlist: frozenset, private: bool) -> list[str]:
    problems = []
    for (path, name), line in sorted(found.items()):
        if (path, name) in allowlist or (name != WHOLE_MODULE and _is_private(name)) != private:
            continue
        if private:
            problems.append(f"{path}:{line} uses private name {name} of {OPS_MODULE}; {REDIRECT}")
        else:
            problems.append(f"{path}:{line} relies on {OPS_MODULE} ({name}); {REDIRECT}")
    return problems


def _stale(found: dict[tuple[str, str], int], allowlist: frozenset) -> list[str]:
    return [
        f"{path} no longer uses {name} of {OPS_MODULE}; remove it from the allowlist"
        for path, name in sorted(allowlist - found.keys())
    ]


@pytest.fixture(scope="module")
def source_usages(source_index) -> dict[tuple[str, str], int]:
    return _scan(SOURCE, source_index)


def test_no_new_private_ops_names(source_usages):
    problems = _unlisted(source_usages, ALLOWLIST, private=True)
    assert not problems, "private names of the vendored ops module:\n" + "\n".join(problems)


def test_no_new_ops_reliance(source_usages):
    problems = _unlisted(source_usages, ALLOWLIST, private=False)
    assert not problems, "new reliance on the vendored ops module:\n" + "\n".join(problems)


def test_allowlist_only_shrinks(source_usages):
    problems = _stale(source_usages, ALLOWLIST)
    assert not problems, "stale allowlist entries:\n" + "\n".join(problems)


def _write(base: Path, relative: str, text: str) -> None:
    target = base / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _names(source: str) -> set[str]:
    return {name for name, _ in _usages(ast.parse(source))}


def test_from_import_records_every_name_including_lazy_ones():
    source = (
        "from vendor.gpl.comfyui.ops import pick_operations, _parse_comfy_quant as parse\n"
        "def later():\n"
        "    from vendor.gpl.comfyui.ops import cast_to\n"
    )

    assert _names(source) == {"pick_operations", "_parse_comfy_quant", "cast_to"}


@pytest.mark.parametrize("source", [
    "from vendor.gpl.comfyui import ops\nops.pick_operations()\n",
    "from vendor.gpl.comfyui import ops as a\na.pick_operations()\n",
    "import vendor.gpl.comfyui.ops as a\na.pick_operations()\n",
    "from vendor.gpl import comfyui\ncomfyui.ops.pick_operations()\n",
    "import vendor.gpl.comfyui as c\nc.ops.pick_operations()\n",
], ids=["from-import", "from-import-aliased", "import-aliased", "package-attribute", "package-aliased-attribute"])
def test_each_module_spelling_records_the_module_alone(source):
    assert _names(source) == {WHOLE_MODULE}


@pytest.mark.parametrize("source", [
    "from vendor.gpl.comfyui import ops\nops._hidden()\n",
    "from vendor.gpl.comfyui import ops as a\na._hidden()\n",
    "import vendor.gpl.comfyui.ops as a\na._hidden()\n",
    "import vendor.gpl.comfyui.ops\nvendor.gpl.comfyui.ops._hidden()\n",
    "from vendor.gpl import comfyui\ncomfyui.ops._hidden()\n",
], ids=["from-import", "from-import-aliased", "import-aliased", "import-plain", "package-attribute"])
def test_each_module_spelling_records_private_attributes(source):
    assert "_hidden" in _names(source)


def test_public_and_dunder_attributes_of_an_alias_are_not_private_names():
    source = "import vendor.gpl.comfyui.ops as a\na.pick_operations\na.__name__\n"

    assert _names(source) == {WHOLE_MODULE}


def test_string_spellings_of_the_module_count():
    source = (
        "import importlib, sys\n"
        "NAME = 'vendor.gpl.comfyui.ops'\n"
        "importlib.import_module('vendor.gpl.comfyui.ops')\n"
        "sys.modules.get(NAME)\n"
    )

    assert [line for _, line in _usages(ast.parse(source))] == [2, 3]
    assert _names(source) == {WHOLE_MODULE}


def test_type_checking_imports_are_exempt_but_their_else_branch_is_not():
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from vendor.gpl.comfyui.ops import typed_only\n"
        "else:\n"
        "    from vendor.gpl.comfyui.ops import runtime_name\n"
    )

    assert _names(source) == {"runtime_name"}


def test_sibling_modules_and_relative_imports_do_not_count():
    source = (
        "from vendor.gpl.comfyui.flux.layers import EmbedND\n"
        "from vendor.gpl.comfyui import flux\n"
        "from .ops import local_name\n"
        "from .vendor.gpl.comfyui.ops import sibling_package_name\n"
        "from ...ops.embedding import sinusoidal_embedding\n"
        "flux._private()\n"
    )

    assert _names(source) == set()


def test_scan_skips_the_kernels_package_and_keys_by_repo_relative_path(tmp_path, source_index):
    base = tmp_path / "src"
    _write(base, "features/user.py", "x = 1\nfrom vendor.gpl.comfyui.ops import _hidden\n")
    _write(base, "platform/runtime/native/kernels/impl.py", "from vendor.gpl.comfyui.ops import _hidden\n")
    _write(base, "platform/runtime/native/other.py", "from vendor.gpl.comfyui.ops import fine\n")

    assert _scan(base, source_index) == {
        ("src/features/user.py", "_hidden"): 2,
        ("src/platform/runtime/native/other.py", "fine"): 1,
    }


def test_unlisted_usage_names_file_line_name_and_the_kernels_package(tmp_path, source_index):
    base = tmp_path / "src"
    _write(base, "features/user.py", "x = 1\nfrom vendor.gpl.comfyui.ops import _hidden, shown\n")
    found = _scan(base, source_index)

    private = _unlisted(found, frozenset(), private=True)
    public = _unlisted(found, frozenset(), private=False)

    assert len(private) == 1 and len(public) == 1
    assert "src/features/user.py:2" in private[0] and "private name _hidden" in private[0]
    assert "src/features/user.py:2" in public[0] and "(shown)" in public[0]
    assert "src.platform.runtime.native.kernels" in private[0] + public[0]


def test_listed_usage_passes_and_an_unused_entry_is_stale(tmp_path, source_index):
    base = tmp_path / "src"
    _write(base, "features/user.py", "from vendor.gpl.comfyui.ops import _hidden\n")
    found = _scan(base, source_index)
    allowlist = frozenset({("src/features/user.py", "_hidden"), ("src/features/gone.py", "cast_to")})

    assert _unlisted(found, allowlist, private=True) == []
    assert _unlisted(found, allowlist, private=False) == []
    assert _stale(found, allowlist) == [
        f"src/features/gone.py no longer uses cast_to of {OPS_MODULE}; remove it from the allowlist"
    ]


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


def _capability_references(tree: ast.Module) -> list[int]:
    return _references(tree, CAPABILITY_PROBE)


def _scan_capability_references(base: Path, source_index) -> dict[str, int]:
    found: dict[str, int] = {}
    for relative, tree in _walk_outside_kernels(base, source_index):
        lines = _capability_references(tree)
        if lines:
            found[relative] = lines[0]
    return found


def _unlisted_capability_references(found: dict[str, int], allowlist: frozenset) -> list[str]:
    return [
        f"{path}:{line} references {CAPABILITY_PROBE}; {REDIRECT}"
        for path, line in sorted(found.items())
        if path not in allowlist
    ]


def _stale_capability_entries(found: dict[str, int], allowlist: frozenset) -> list[str]:
    return [
        f"{path} no longer references {CAPABILITY_PROBE}; remove it from the capability allowlist"
        for path in sorted(allowlist - found.keys())
    ]


@pytest.fixture(scope="module")
def capability_references(source_index) -> dict[str, int]:
    return _scan_capability_references(SOURCE, source_index)


def test_no_new_capability_probes_outside_the_kernels_package(capability_references):
    problems = _unlisted_capability_references(capability_references, CAPABILITY_ALLOWLIST)
    assert not problems, "device capability probed outside the kernels package:\n" + "\n".join(problems)


def test_capability_allowlist_only_shrinks(capability_references):
    problems = _stale_capability_entries(capability_references, CAPABILITY_ALLOWLIST)
    assert not problems, "stale capability allowlist entries:\n" + "\n".join(problems)


@pytest.mark.parametrize("source, lines", [
    ("import torch\ntorch.cuda.get_device_capability()\n", [2]),
    ("from torch.cuda import get_device_capability\nget_device_capability()\n", [1, 2]),
    ("x = [torch.cuda.get_device_capability(i) for i in range(2)]\n", [1]),
    ("import torch\nprobe = torch.cuda.get_device_capability\n", [2]),
], ids=["attribute-call", "bare-name-call", "comprehension", "bare-reference"])
def test_every_reference_to_the_capability_probe_is_recorded(source, lines):
    assert _capability_references(ast.parse(source)) == lines


@pytest.mark.parametrize("source", [
    "import torch\nprobe = getattr(torch.cuda, 'get_device_capability')\n",
    "from torch.cuda import get_device_capability as probe\n",
], ids=["string-constant", "aliased-import"])
def test_string_and_import_spellings_of_the_capability_probe_are_recorded(source):
    assert len(_capability_references(ast.parse(source))) == 1


@pytest.mark.parametrize("source", [
    "import torch\ntorch.cuda.get_device_name(0)\n",
    "def get_device_capability_text():\n    return 'torch.cuda.get_device_capability()'\n",
    '"""torch.cuda.get_device_capability() reads the compute capability"""\n',
    "from src.platform.runtime.native.kernels import capabilities\ncapabilities().device(0).capability\n",
], ids=["device-name", "string-and-lookalike-name", "docstring-sentence", "snapshot-read"])
def test_other_device_queries_and_sentences_are_not_capability_references(source):
    assert _capability_references(ast.parse(source)) == []


def test_capability_references_are_listed_by_line_in_order():
    source = "import torch\nx = 1\ntorch.cuda.get_device_capability()\ny = torch.cuda.get_device_capability(1)\n"

    assert _capability_references(ast.parse(source)) == [3, 4]


def test_unlisted_capability_reference_names_file_line_and_the_kernels_package(tmp_path, source_index):
    base = tmp_path / "src"
    _write(base, "features/user.py", "import torch\nx = 1\ntorch.cuda.get_device_capability()\n")

    problems = _unlisted_capability_references(_scan_capability_references(base, source_index), frozenset())

    assert len(problems) == 1
    assert "src/features/user.py:3" in problems[0]
    assert "get_device_capability" in problems[0]
    assert "src.platform.runtime.native.kernels" in problems[0]


def test_listed_capability_reference_passes_and_an_unused_entry_is_stale(tmp_path, source_index):
    base = tmp_path / "src"
    _write(base, "features/user.py", "import torch\ntorch.cuda.get_device_capability()\n")
    found = _scan_capability_references(base, source_index)
    allowlist = frozenset({"src/features/user.py", "src/features/gone.py"})

    assert _unlisted_capability_references(found, allowlist) == []
    assert _stale_capability_entries(found, allowlist) == [
        "src/features/gone.py no longer references get_device_capability; remove it from the capability allowlist"
    ]


def _scan_direct_attention(base: Path, source_index) -> dict[str, int]:
    found: dict[str, int] = {}
    for relative, tree in _walk_outside_kernels(base, source_index):
        lines = _references(tree, DIRECT_ATTENTION)
        if lines:
            found[relative] = lines[0]
    return found


def _unlisted_direct_attention(found: dict[str, int], allowlist: dict[str, str]) -> list[str]:
    return [
        f"{path}:{line} references {DIRECT_ATTENTION}; {REDIRECT} (attention)"
        for path, line in sorted(found.items())
        if path not in allowlist
    ]


def _stale_direct_attention(found: dict[str, int], allowlist: dict[str, str]) -> list[str]:
    return [
        f"{path} no longer references {DIRECT_ATTENTION}; remove it from the direct attention allowlist"
        for path in sorted(allowlist.keys() - found.keys())
    ]


@pytest.fixture(scope="module")
def direct_attention(source_index) -> dict[str, int]:
    return _scan_direct_attention(SOURCE, source_index)


def test_no_new_direct_attention_callers_outside_the_kernels_package(direct_attention):
    problems = _unlisted_direct_attention(direct_attention, DIRECT_ATTENTION_ALLOWLIST)
    assert not problems, "scaled dot product attention called outside the kernels package:\n" + "\n".join(problems)


def test_direct_attention_allowlist_only_shrinks(direct_attention):
    problems = _stale_direct_attention(direct_attention, DIRECT_ATTENTION_ALLOWLIST)
    assert not problems, "stale direct attention allowlist entries:\n" + "\n".join(problems)


def test_every_direct_attention_entry_gives_a_one_line_reason():
    for path, reason in DIRECT_ATTENTION_ALLOWLIST.items():
        assert path.startswith("src/") and path.endswith(".py"), path
        assert reason.strip() and "\n" not in reason, path


@pytest.mark.parametrize("source, lines", [
    ("import torch.nn.functional as F\nout = F.scaled_dot_product_attention(q, k, v)\n", [2]),
    ("import torch\nout = torch.nn.functional.scaled_dot_product_attention(q, k, v)\n", [2]),
    ("import torch.nn.functional as F\nrescue = F.scaled_dot_product_attention\n", [2]),
    ("from torch.nn.functional import scaled_dot_product_attention\nscaled_dot_product_attention(q, k, v)\n", [1, 2]),
    ("import torch.nn.functional as F\nout = [F.scaled_dot_product_attention(q, k, v) for q in qs]\n", [2]),
], ids=["attribute-call", "fully-qualified", "bare-reference", "imported-name", "comprehension"])
def test_every_reference_to_direct_attention_is_recorded(source, lines):
    assert _references(ast.parse(source), DIRECT_ATTENTION) == lines


@pytest.mark.parametrize("source", [
    "import torch.nn.functional as F\nrun = getattr(F, 'scaled_dot_product_attention')\n",
    "from torch.nn.functional import scaled_dot_product_attention as sdpa\nsdpa(q, k, v)\n",
], ids=["string-constant", "aliased-import"])
def test_string_and_import_spellings_of_direct_attention_are_recorded(source):
    assert len(_references(ast.parse(source), DIRECT_ATTENTION)) == 1


@pytest.mark.parametrize("source", [
    "from ..kernels import attention\nattention(q, k, v)\n",
    "from .attention import sparse_scaled_dot_product_attention\nsparse_scaled_dot_product_attention(q, k, v)\n",
    '"""F.scaled_dot_product_attention is the plain pytorch call"""\n',
    "def plain_scaled_dot_product_attention_text():\n    return 'uses F.scaled_dot_product_attention'\n",
], ids=["dispatcher-call", "sparse-lookalike", "docstring-sentence", "string-and-lookalike-name"])
def test_dispatcher_calls_and_sentences_are_not_direct_attention(source):
    assert _references(ast.parse(source), DIRECT_ATTENTION) == []


def test_direct_attention_scan_skips_the_kernels_package_and_keys_by_repo_relative_path(tmp_path, source_index):
    base = tmp_path / "src"
    call = "import torch.nn.functional as F\nx = 1\nF.scaled_dot_product_attention(q, k, v)\n"
    _write(base, "features/user.py", call)
    _write(base, "platform/runtime/native/kernels/impl.py", call)
    _write(base, "platform/runtime/native/vae/block.py", call)
    _write(base, "platform/runtime/native/vae/clean.py", "x = 1\n")

    assert _scan_direct_attention(base, source_index) == {
        "src/features/user.py": 3,
        "src/platform/runtime/native/vae/block.py": 3,
    }


def test_unlisted_direct_attention_names_file_line_and_the_kernels_package(tmp_path, source_index):
    base = tmp_path / "src"
    _write(base, "features/user.py", "import torch.nn.functional as F\nx = 1\nF.scaled_dot_product_attention(q, k, v)\n")

    problems = _unlisted_direct_attention(_scan_direct_attention(base, source_index), {})

    assert len(problems) == 1
    assert "src/features/user.py:3" in problems[0]
    assert DIRECT_ATTENTION in problems[0]
    assert "src.platform.runtime.native.kernels" in problems[0]


def test_listed_direct_attention_passes_and_an_unused_entry_is_stale(tmp_path, source_index):
    base = tmp_path / "src"
    _write(base, "features/user.py", "import torch.nn.functional as F\nF.scaled_dot_product_attention(q, k, v)\n")
    found = _scan_direct_attention(base, source_index)
    allowlist = {"src/features/user.py": "kept on purpose", "src/features/gone.py": "kept on purpose"}

    assert _unlisted_direct_attention(found, allowlist) == []
    assert _stale_direct_attention(found, allowlist) == [
        "src/features/gone.py no longer references scaled_dot_product_attention; "
        "remove it from the direct attention allowlist"
    ]
