"""Lazy, thread-safe construction of a shared ``chromadb`` client.

``chromadb`` and its ``PersistentClient`` are heavy to import (telemetry,
OpenTelemetry exporters, ...), so the import and the client construction are
both deferred to first use rather than paid at process boot. A
``ChromaClientProvider`` owns the lifecycle of exactly one client for one
``persist_dir``; construct it once and inject the same instance into every
vector store that persists to that directory so they open one underlying
client instead of each opening (and holding a connection to) its own.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import chromadb

logger = logging.getLogger(__name__)


class ChromaClientProvider:
    """Builds and caches one ``chromadb.PersistentClient`` on first use."""

    def __init__(self, persist_dir: str):
        self._persist_dir = persist_dir
        self._client: Optional["chromadb.ClientAPI"] = None
        self._lock = threading.Lock()

    def get(self) -> "chromadb.ClientAPI":
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import chromadb

                    self._client = chromadb.PersistentClient(path=self._persist_dir)
        return self._client

    def close(self) -> None:
        """Release the cached client so the next ``get()`` builds a fresh one.

        Idempotent, and safe to call whether or not a client has been built
        yet. A failed ``get()`` never caches anything, so a later ``get()``
        call already retries on its own; ``close()`` is for a caller that
        holds a live client and wants it released deliberately.

        This provider is the only place a live client may be cached - callers
        (the vector stores) must call ``get()`` on every access rather than
        holding their own reference, or a store that cached an earlier result
        keeps serving a client this provider has since closed. ``close()``
        assumes a quiescent boundary: call it only when nothing else is
        concurrently reading the client returned by a still-in-flight
        ``get()`` call, since closing races with an in-progress read of the
        about-to-be-replaced client are the caller's to avoid, not this
        provider's to prevent.
        """
        with self._lock:
            client, self._client = self._client, None
        if client is not None:
            client.close()


class ChromaCollectionStore:
    collection_prefix: str = ""

    def __init__(
        self,
        persist_dir: str = "storage/chromadb",
        embedder_slug: str = "default",
        client_provider: Optional[ChromaClientProvider] = None,
    ):
        self._persist_dir = persist_dir
        self._embedder_slug = embedder_slug
        self._client: Optional["chromadb.ClientAPI"] = None
        self._client_provider = client_provider or ChromaClientProvider(persist_dir)

    @property
    def client(self) -> "chromadb.ClientAPI":
        if self._client is not None:
            return self._client
        return self._client_provider.get()

    def _collection_name(self, user_id: str) -> str:
        return f"{self.collection_prefix}{user_id}__{self._embedder_slug}"

    def get_collection(self, user_id: str) -> "chromadb.Collection":
        return self.client.get_or_create_collection(
            name=self._collection_name(user_id),
            metadata={"hnsw:space": "cosine"},
        )

    def is_collection_empty(self, user_id: str) -> bool:
        try:
            return self.get_collection(user_id).count() == 0
        except Exception as e:
            logger.error(f"ChromaDB is_collection_empty failed on '{self.collection_prefix}{user_id}': {e}")
            return False

    def bulk_delete(self, user_id: str, ids: List[str]) -> None:
        if not ids:
            return
        collection = self.get_collection(user_id)
        try:
            collection.delete(ids=ids)
        except Exception as e:
            logger.error(f"ChromaDB bulk_delete failed on '{self.collection_prefix}{user_id}': {e}")
