import asyncio
from typing import Any, Dict, List, Optional

from src.features.llm.tools.base import BaseTool, ToolApprovalPreview, ToolContext, ToolResult


class Probe(BaseTool):
    description = "A probe."
    parameters = {"type": "object", "properties": {}}

    def __init__(
        self,
        name: str,
        log: List[str],
        *,
        read_only: bool = False,
        gated: bool = False,
        shared: Optional["Shared"] = None,
        image: Optional[str] = None,
    ) -> None:
        self.name = name
        self.read_only = read_only
        self.requires_approval = gated
        self._log = log
        self._shared = shared or Shared()
        self._image = image

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        shared = self._shared
        self._log.append(f"start:{self.name}")
        shared.running += 1
        shared.peak = max(shared.peak, shared.running)
        try:
            if self.name in shared.entered:
                shared.entered[self.name].set()
            if shared.barrier is not None and self.read_only:
                await shared.barrier.wait()
            if self.name in shared.hold:
                await shared.hold[self.name].wait()
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            shared.cancelled.append(self.name)
            raise
        finally:
            shared.running -= 1
        self._log.append(f"end:{self.name}")
        if self.requires_approval:
            return ToolResult(success=True, data="", preview=ToolApprovalPreview(action="Do", target=self.name))
        return ToolResult(success=True, data=f"{self.name} done", image_data=self._image)

    async def execute_confirmed(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        self._log.append(f"confirm:{self.name}")
        await asyncio.sleep(0)
        return ToolResult(success=True, data=f"{self.name} applied")


class Shared:
    def __init__(self) -> None:
        self.running = 0
        self.peak = 0
        self.hold: Dict[str, asyncio.Event] = {}
        self.entered: Dict[str, asyncio.Event] = {}
        self.barrier: Optional[asyncio.Barrier] = None
        self.cancelled: List[str] = []
