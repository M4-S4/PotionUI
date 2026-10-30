"""Setup-status DTO.

Deliberately minimal: the setup-status endpoint is public and unauthenticated,
so it exposes only what a first-run client needs to route on and nothing about
the host (no hardware, paths, versions, or user details).
"""

from typing import Optional

from pydantic import BaseModel


class SetupStatus(BaseModel):
    """Public first-run status. These three booleans are the entire contract."""

    needs_owner: bool
    registration_open: bool
    claim_requires_token: bool


class MachineGpu(BaseModel):
    name: Optional[str] = None
    vram_bytes: int
    compute_capability: Optional[str] = None
    supported: bool


class ModelsDisk(BaseModel):
    path: str
    free_bytes: int
    total_bytes: int


class MachineInfo(BaseModel):
    """Admin-only host facts for the setup wizard; each part is null when its probe fails."""

    gpu: Optional[MachineGpu] = None
    ram_total_bytes: Optional[int] = None
    models_disk: Optional[ModelsDisk] = None
