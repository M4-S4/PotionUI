"""
Automation WebSocket Connection Manager.

Handles connections for the `/ws/automations` live run-status channel.
Unlike the per-user notification channel, automation run updates are
broadcast to every authenticated connection (there is no per-user
filtering) - keyed by `client_id` rather than `user_id`. Provides the
same sync-to-async bridge
(`schedule_send`) as `NotificationConnectionHub` so the engine's
`emit_ws` callback (itself already async) and any worker-thread callers can
push messages without awaiting a coroutine directly.
"""
import asyncio
import json
import logging
from typing import Set
from fastapi import WebSocket

from src.platform.security.failure_text import GENERIC_FAILURE, scrub_paths
from src.platform.websocket.base_connection_hub import BaseConnectionHub
from src.platform.websocket.loop_dispatch import LoopDispatchMixin

logger = logging.getLogger(__name__)


class AutomationConnectionHub(BaseConnectionHub, LoopDispatchMixin):
    """Manages WebSocket connections for the automation run-status channel."""

    _DISPATCH_LABEL = "automation send"

    _CONNECTION_LABEL = "Automation client"

    def __init__(self):
        super().__init__()
        self._admin_clients: Set[str] = set()
        self._init_loop_dispatch()

    async def connect(self, websocket: WebSocket, client_id: str, is_admin: bool = False) -> bool:
        """Accept a WebSocket connection and register it."""
        if not await super().connect(websocket, client_id):
            return False
        if is_admin:
            self._admin_clients.add(client_id)
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return True

    def disconnect(self, client_id: str) -> None:
        """Remove a client connection."""
        super().disconnect(client_id)
        self._admin_clients.discard(client_id)

    @staticmethod
    def _for_audience(message: dict, is_admin: bool) -> dict:
        error = message.get("error")
        if not isinstance(error, str):
            return message
        return {**message, "error": scrub_paths(error) if is_admin else GENERIC_FAILURE}

    async def broadcast(self, message: dict) -> None:
        """Send a message to every connected client."""
        try:
            admin_payload = self._for_audience(message, True)
            public_payload = self._for_audience(message, False)
            admin_message = json.dumps(admin_payload)
            public_message = admin_message if public_payload is admin_payload else json.dumps(public_payload)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize automation message: {e}")
            return

        await self._send_all(
            self.active_connections,
            lambda client_id: admin_message if client_id in self._admin_clients else public_message,
            "send automation message",
        )

    def schedule_send(self, message: dict) -> None:
        """Sync-to-async bridge: schedule a broadcast from non-async code."""
        self._schedule(lambda: self.broadcast(message))


# Global instance
automation_connection_hub = AutomationConnectionHub()
