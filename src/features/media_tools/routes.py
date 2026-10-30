import asyncio
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, Depends, Query, Response

from src.features.media_tools.dto import (
    CatalogResponse,
    PreviewRequest,
    StartRunRequest,
    StartRunResponse,
)
from src.features.media_tools.service import MediaToolService
from src.platform.http.base_controller import APIResponse, BaseController
from src.platform.http.error_pipeline import Fail, guarded
from src.platform.media_tools import FfmpegUnavailable, ToolFailed
from src.platform.security.current_user import get_current_active_user

if TYPE_CHECKING:
    from src.bootstrap.container import AppContainer

_TOOL_ERRORS = {
    FfmpegUnavailable: Fail("ffmpeg_unavailable", status=500),
    ToolFailed: Fail("tool_failed", status=500),
}


class MediaToolController(BaseController):

    def __init__(self, service: MediaToolService):
        super().__init__()
        self.service = service

    @guarded(_TOOL_ERRORS, Fail("media_tools_failed", "Failed to list media tools", 500))
    async def catalog(self, kind: Optional[str], scope: Optional[str], current_user) -> APIResponse:
        tools = await asyncio.to_thread(self.service.catalog, current_user, kind, scope)
        return self.success_response(data=CatalogResponse(tools=tools).model_dump())

    @guarded(_TOOL_ERRORS, Fail("media_tool_run_failed", "Failed to start the media tool run", 500))
    async def start_run(self, request: StartRunRequest, current_user) -> APIResponse:
        run_id = await asyncio.to_thread(
            self.service.start_run, current_user, request.steps, request.inputs, request.output
        )
        return self.success_response(data=StartRunResponse(run_id=run_id).model_dump())

    @guarded(_TOOL_ERRORS, Fail("media_tool_run_failed", "Failed to read the media tool run", 500))
    async def get_run(self, run_id: str, current_user) -> APIResponse:
        run = await asyncio.to_thread(self.service.get_run, current_user, run_id)
        return self.success_response(data=run.model_dump())

    @guarded(_TOOL_ERRORS, Fail("media_tool_run_failed", "Failed to cancel the media tool run", 500))
    async def cancel_run(self, run_id: str, current_user) -> APIResponse:
        run = await asyncio.to_thread(self.service.cancel_run, current_user, run_id)
        return self.success_response(data=run.model_dump())

    @guarded(_TOOL_ERRORS, Fail("media_tool_preview_failed", "Failed to render the preview", 500))
    async def preview(self, request: PreviewRequest, current_user) -> Response:
        data = await asyncio.to_thread(
            self.service.preview,
            current_user,
            request.tool,
            request.params,
            request.input,
            request.time_seconds,
        )
        return Response(content=data, media_type="image/webp", headers={"Cache-Control": "no-store"})


def build_router(container: "AppContainer") -> APIRouter:
    controller = container.media_tool_controller
    router = APIRouter(tags=["Media Tools"])

    @router.get("/api/media-tools", response_model=APIResponse, summary="List Media Tools")
    async def list_media_tools(
        kind: Optional[str] = Query(None),
        scope: Optional[str] = Query(None),
        current_user=Depends(get_current_active_user),
    ) -> APIResponse:
        return await controller.catalog(kind, scope, current_user)

    @router.post(
        "/api/media-tools/runs", response_model=APIResponse, status_code=202, summary="Start A Media Tool Run"
    )
    async def start_media_tool_run(
        request: StartRunRequest, current_user=Depends(get_current_active_user)
    ) -> APIResponse:
        return await controller.start_run(request, current_user)

    @router.get("/api/media-tools/runs/{run_id}", response_model=APIResponse, summary="Get A Media Tool Run")
    async def get_media_tool_run(
        run_id: str, current_user=Depends(get_current_active_user)
    ) -> APIResponse:
        return await controller.get_run(run_id, current_user)

    @router.delete(
        "/api/media-tools/runs/{run_id}", response_model=APIResponse, summary="Cancel A Media Tool Run"
    )
    async def cancel_media_tool_run(
        run_id: str, current_user=Depends(get_current_active_user)
    ) -> APIResponse:
        return await controller.cancel_run(run_id, current_user)

    @router.post("/api/media-tools/preview", response_model=None, summary="Preview A Media Tool")
    async def preview_media_tool(
        request: PreviewRequest, current_user=Depends(get_current_active_user)
    ) -> Response:
        return await controller.preview(request, current_user)

    return router
