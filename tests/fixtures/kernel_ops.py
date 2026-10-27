from __future__ import annotations

import importlib
import sys

KERNELS_PACKAGE = "src.platform.runtime.native.kernels"


def patch_ops_name(monkeypatch, name: str, value) -> None:
    import vendor.gpl.comfyui.ops as ops

    importlib.import_module(KERNELS_PACKAGE)
    modules = [ops] + [
        module
        for key, module in sorted(sys.modules.items())
        if module is not None and (key == KERNELS_PACKAGE or key.startswith(KERNELS_PACKAGE + "."))
    ]
    patched = 0
    for module in modules:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)
            patched += 1
    assert patched, f"{name} is defined in none of the modules that hold it"
