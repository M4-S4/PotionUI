"""ChromaDB-backed vector storage for gallery generation-prompt embeddings.

A third vector space alongside `GalleryVectorStore` (SigLIP image-content
embeddings) and `PromptVectorStore` (the user-authored prompt library):
this one embeds the actual text prompt a file was generated from, using the
same text embedder that backs prompt-database search
(`src.features.prompt_database.embedding`). Namespaced under its own
``gallery_prompts_`` collection prefix - distinct from both
``gallery_`` (image embeddings) and ``rich_prompts_`` (the prompt library) -
so neither an embedder switch nor either sibling store's stale-collection
pruning can ever touch or mix into this one.

Lazy client construction is delegated to a shared ``ChromaClientProvider``
(``src.platform.vector.chroma_client``), the same instance injected into
``GalleryVectorStore`` and ``PromptVectorStore`` when they persist to the
same directory.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.platform.vector.chroma_client import ChromaCollectionStore

logger = logging.getLogger(__name__)


class GalleryPromptVectorStore(ChromaCollectionStore):
    collection_prefix = "gallery_prompts_"

    def add(
        self,
        user_id: str,
        file_id: str,
        embedding: List[float],
        prompt_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upsert one file's prompt-text embedding into the user's collection."""
        collection = self.get_collection(user_id)
        meta = metadata or {}
        safe_meta = {
            k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))
        }
        collection.upsert(
            ids=[file_id],
            embeddings=[embedding],
            documents=[prompt_text],
            metadatas=[safe_meta] if safe_meta else None,
        )
