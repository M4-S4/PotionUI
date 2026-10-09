import importlib
import struct
import sys
import zlib
from pathlib import Path


def _tiny_png() -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff\xff")) + chunk(b"IEND", b"")


TINY_PNG = _tiny_png()


def bind_plugin_backend(plugin_root: Path) -> None:
    for key in list(sys.modules):
        if key == "backend" or key.startswith("backend."):
            del sys.modules[key]
    if str(plugin_root) in sys.path:
        sys.path.remove(str(plugin_root))
    sys.path.insert(0, str(plugin_root))
    importlib.invalidate_caches()
    importlib.import_module("backend")
