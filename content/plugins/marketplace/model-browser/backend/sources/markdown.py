import hashlib
import html
import logging
import re
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

logger = logging.getLogger("plugins.model-browser")

T = TypeVar("T")

MARKDOWN_EXTRAS: Dict[str, Any] = {
    "fenced-code-blocks": None,
    "highlightjs-lang": None,
    "tables": None,
    "strike": None,
    "cuddled-lists": None,
    "middle-word-em": False,
}

PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


def paragraphs_html(text: str) -> str:
    return "".join(f"<p>{html.escape(part.strip())}</p>" for part in PARAGRAPH_BREAK.split(text) if part.strip())


def convert_markdown(text: str, failure: str) -> Optional[str]:
    try:
        import markdown2
    except ImportError:
        return None
    try:
        return str(markdown2.markdown(text, extras=dict(MARKDOWN_EXTRAS)))
    except Exception:
        logger.warning("[MODEL-BROWSER] %s; showing it as text", failure, exc_info=True)
        return None


class RenderMemo(Generic[T]):
    def __init__(self, size: int) -> None:
        self._size = size
        self._entries: "OrderedDict[str, T]" = OrderedDict()
        self._lock = threading.Lock()

    def render(self, text: str, render: Callable[[str], T]) -> T:
        key = hashlib.md5(text.encode("utf-8", "replace"), usedforsecurity=False).hexdigest()
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None:
                self._entries.move_to_end(key)
                return hit
        value = render(text)
        with self._lock:
            self._entries[key] = value
            while len(self._entries) > self._size:
                self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
