from __future__ import annotations

def patch_ops_name(monkeypatch, name: str, value) -> None:
    import vendor.gpl.comfyui.ops as ops
    from src.platform.runtime.native.kernels import fp8_linear, nvfp4_linear, operations

    patched = 0
    for module in (ops, fp8_linear, nvfp4_linear, operations):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)
            patched += 1
    assert patched, f"{name} is defined in none of the modules that hold it"
