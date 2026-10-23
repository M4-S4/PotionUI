"""Resolve a recipe artifact to the index row that backs it."""

from typing import Any, Optional, Tuple


def find_artifact_model(
    model_repository: Any,
    artifact: Any,
    type_manager: Optional[Any] = None,
    digests: Optional[Any] = None,
    hash_candidates: str = "never",
) -> Optional[Any]:
    model = model_repository.get_by_identity(artifact.model_type, artifact.filename)
    if model is not None:
        return model
    checksum = getattr(artifact, "checksum", None)
    sha256 = getattr(checksum, "value", None) if checksum else None
    lookup = getattr(model_repository, "get_by_sha256", None)
    if not sha256 or lookup is None:
        return None
    by_hash = lookup(sha256)
    size = getattr(artifact, "size_bytes", None)
    if by_hash is None and digests is not None and size:
        by_hash = digests.find_by_digest(sha256, size, hash_candidates).model
    if by_hash is None or not getattr(by_hash, "is_available", True):
        return None
    if by_hash.model_type != artifact.model_type and type_manager is not None:
        type_manager.assert_type(sha256, artifact.model_type, "recipe")
        by_hash = lookup(sha256) or by_hash
    return by_hash


def _same_size_match(artifact: Any, digests: Optional[Any]) -> Optional[Any]:
    checksum = getattr(artifact, "checksum", None)
    sha256 = getattr(checksum, "value", None) if checksum else None
    size = getattr(artifact, "size_bytes", None)
    if digests is None or not sha256 or not size:
        return None
    match = digests.find_by_digest(sha256, size, "never")
    return None if match.model is not None else match


def same_size_candidates(artifact: Any, digests: Optional[Any]) -> int:
    match = _same_size_match(artifact, digests)
    return 0 if match is None else match.candidates


def slot_candidate_counts(artifact: Any, digests: Optional[Any]) -> Tuple[int, int]:
    variants = getattr(artifact, "variants", None) or ()
    resolved = [artifact.resolve(variant.id) for variant in variants] if variants else [artifact]
    candidates = waiting = 0
    for item in resolved:
        match = _same_size_match(item, digests)
        if match is not None:
            candidates += match.candidates
            waiting += match.waiting
    return candidates, waiting


def find_slot_model(
    model_repository: Any,
    artifact: Any,
    variant_id: Optional[str] = None,
    type_manager: Optional[Any] = None,
    digests: Optional[Any] = None,
    hash_candidates: str = "never",
) -> Optional[Any]:
    variants = getattr(artifact, "variants", None) or ()
    if not variants:
        return find_artifact_model(model_repository, artifact, type_manager, digests, hash_candidates)
    ordered = [v.id for v in variants]
    if variant_id in ordered:
        ordered.remove(variant_id)
        ordered.insert(0, variant_id)
    for candidate in ordered:
        model = find_artifact_model(
            model_repository, artifact.resolve(candidate), type_manager, digests, hash_candidates
        )
        if model is not None:
            return model
    return None
