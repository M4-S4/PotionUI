"""
Notification WebSocket Connection Manager.

Handles per-user WebSocket connections for the notification bell/panel,
keyed by `user_id -> {client_id: WebSocket}` to support multiple tabs per
user. Also provides a sync-to-async bridge (`schedule_send`) so callers on
worker threads (e.g. generation completion) can push messages without
awaiting a coroutine directly.
"""
import asyncio
import json
import logging
from typing import Dict, Optional, Set
from fastapi import WebSocket

from src.platform.websocket.base_connection_hub import BaseConnectionHub
from src.platform.websocket.loop_dispatch import LoopDispatchMixin

logger = logging.getLogger(__name__)


class NotificationConnectionHub(BaseConnectionHub, LoopDispatchMixin):
    """Manages WebSocket connections for the per-user notification channel."""

    _CONNECTION_LABEL = "Notification client"
    _DISPATCH_LABEL = "notification send"

    def __init__(self):
        super().__init__()
        self._user_clients: Dict[str, Set[str]] = {}
        self._client_users: Dict[str, str] = {}
        self._init_loop_dispatch()

    async def connect(self, websocket: WebSocket, client_id: str, user_id: str) -> bool:
        """Accept a WebSocket connection and register it for the user."""
        if not await super().connect(websocket, client_id):
            return False
        self._user_clients.setdefault(user_id, set()).add(client_id)
        self._client_users[client_id] = user_id
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return True

    def disconnect(self, client_id: str) -> None:
        """Remove a client connection."""
        super().disconnect(client_id)
        user_id = self._client_users.pop(client_id, None)
        user_clients = self._user_clients.get(user_id)
        if user_clients is None:
            return
        user_clients.discard(client_id)
        if not user_clients:
            del self._user_clients[user_id]

    async def send_to_user(self, user_id: str, message: dict) -> None:
        """Send a message to all of a user's connected clients (all tabs)."""
        client_ids = self._user_clients.get(user_id)
        if not client_ids:
            return

        try:
            json_message = json.dumps(message)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize notification message: {e}")
            return

        targets = {
            client_id: self.active_connections[client_id]
            for client_id in client_ids
            if client_id in self.active_connections
        }
        await self._send_all(targets, json_message, "send notification")

    def schedule_send(self, user_id: Optional[str], message: dict) -> None:
        """Sync-to-async bridge: schedule a send from non-async code.
        `user_id=None` broadcasts to all connected users."""
        def _make_coro():
            return self.broadcast(message) if user_id is None else self.send_to_user(user_id, message)

        self._schedule(_make_coro)


# Global instance
notification_connection_hub = NotificationConnectionHub()
