"""
DTOs for model API endpoints.

Request/Response models for model index management operations.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, model_validator


# ============ Request DTOs ============


class ModelInfoFetchRequest(BaseModel):
    """Request to fetch model info from marketplace provider"""
    model_ids: Optional[List[str]] = None
    provider: str  # Required - no default, must be specified
    force_refresh: Optional[bool] = False
    wait: Optional[bool] = False


class UpdateDescriptionRequest(BaseModel):
    """Request to update model description"""
    description: str


class SetModelTypeRequest(BaseModel):
    model_type: str


class IndexingSettingsRequest(BaseModel):
    level: Optional[Literal["low", "normal", "max"]] = None
    policy: Optional[Literal["unset", "background", "on_demand"]] = None


class IndexingDigestsRequest(BaseModel):
    mode: Literal["all", "selected", "matching"]
    model_ids: Optional[List[str]] = None
    query: Optional[str] = None
    model_type: Optional[str] = None
    exclude_model_ids: Optional[List[str]] = None

    @model_validator(mode="after")
    def _selected_needs_model_ids(self) -> "IndexingDigestsRequest":
        if self.mode == "selected" and self.model_ids is None:
            raise ValueError("mode 'selected' needs model_ids")
        return self


class UpdatePromptingGuidanceRequest(BaseModel):
    """Request to update a model's admin-authored prompting guidance"""
    prompting_guidance: str


class UpdateModelMetadataRequest(BaseModel):
    """Request to update a model's shared attribute values (e.g. a LoRA's strength)"""
    values: Dict[str, Any]


class UpdateModelUserAttributesRequest(BaseModel):
    """Request to update the caller's per-user attribute value overlay for a model.
    Every key must name a `per_user` attribute definition."""
    values: Dict[str, Any]


class CreateAttributeDefinitionRequest(BaseModel):
    """Request to create a model attribute definition (admin only)."""
    key: str
    label: str
    field_type: str
    model_types: List[str] = []
    config: Dict[str, Any] = {}
    default_value: Optional[Any] = None
    description: Optional[str] = None
    per_user: bool = False
    admin_only: bool = False


class UpdateAttributeDefinitionRequest(BaseModel):
    """Request to update a model attribute definition (admin only). Fields left
    unset are unchanged; on a system definition, `key`/`field_type` are
    immutable."""
    key: Optional[str] = None
    label: Optional[str] = None
    field_type: Optional[str] = None
    model_types: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    default_value: Optional[Any] = None
    description: Optional[str] = None
    per_user: Optional[bool] = None
    admin_only: Optional[bool] = None


class ModelPreviewInput(BaseModel):
    """A model preview to set, referencing a file already in storage.

    `source_path` is a storage-relative path to a file uploaded through the media
    feature (`/api/media/upload`) or picked from generation history. The backend
    registers it as a `files` row and serves the preview via the auth-exempt
    `/api/media/files/<id>` route so it renders in plain `<img>`/`<video>` tags.
    """
    source_path: str
    type: Literal['image', 'video', 'audio']
    name: Optional[str] = None


class UpdateModelPreviewRequest(BaseModel):
    """Request to set (a preview input) or clear (null) a model's preview."""
    preview: Optional[ModelPreviewInput] = None


class AddModelPreviewRequest(BaseModel):
    """Request to append one preview to a model's preview list."""
    preview: ModelPreviewInput


class ReorderModelPreviewsRequest(BaseModel):
    """Request to reorder a model's preview list.

    `ordered_ids` must be exactly the model's current preview ids (from
    `GET .../previews`), in the desired order.
    """
    ordered_ids: List[str]


class UpdateTagsRequest(BaseModel):
    """Request to update model tags"""
    tag_ids: List[str]


class DownloadModelRequest(BaseModel):
    """Request to download and index a model"""
    name: str
    link: str
    size: str
    sha256: str
    model_type: Optional[str] = 'checkpoint'


class RecommendationDownloadRequest(BaseModel):
    """Request to download a `model` field recommendation (v2 - provider-gated).

    Either `provider` + `ref` (provider-backed - `ref` is an OPAQUE, provider-native
    string; core only attempts a `{"provider_model_id": ..., "provider_version_id":
    ...}` JSON decode as a convention for resolving a download URL via the provider
    registry, falling back to treating the whole string as `provider_model_id`) or
    `link` (provider-less, today's shape) must be given. See docs/presets.md
    "recommendations".
    """
    name: str
    model_type: str = 'checkpoint'
    provider: Optional[str] = None
    ref: Optional[str] = None
    link: Optional[str] = None
    sha256: Optional[str] = None


class UserModelAssignmentRequest(BaseModel):
    """Request to assign/unassign a model to a user"""
    user_id: str
    model_id: str


class ModelFavoriteRequest(BaseModel):
    """Request to set/clear a model's favorite flag (per requesting user)"""
    is_favorite: bool


class ModelLibraryNameRequest(BaseModel):
    """Request to set/clear a model's per-user custom display name"""
    name: Optional[str] = None