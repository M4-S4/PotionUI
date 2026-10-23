"""Tests for the NotificationConnectionHub class."""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock
from fastapi import WebSocket

from src.platform.websocket.notification_connection_hub import NotificationConnectionHub


class TestNotificationConnectionHub:

    @pytest.fixture
    def manager(self):
        return NotificationConnectionHub()

    @pytest.fixture
    def mock_websocket(self):
        return AsyncMock(spec=WebSocket)

    def test_init(self, manager):
        assert manager.active_connections == {}

    # ========== connect / multi-tab ==========

    @pytest.mark.asyncio
    async def test_connect_accepts_and_registers(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1", "user-1")

        mock_websocket.accept.assert_called_once()
        assert manager.active_connections["client-1"] is mock_websocket

    @pytest.mark.asyncio
    async def test_connect_multi_tab_same_user(self, manager, mock_websocket):
        second_websocket = AsyncMock(spec=WebSocket)

        await manager.connect(mock_websocket, "client-1", "user-1")
        await manager.connect(second_websocket, "client-2", "user-1")

        assert len(manager._user_clients["user-1"]) == 2
        assert manager.active_connections["client-1"] is mock_websocket
        assert manager.active_connections["client-2"] is second_websocket

    # ========== disconnect ==========

    @pytest.mark.asyncio
    async def test_disconnect_removes_client(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1", "user-1")

        manager.disconnect("client-1")

        assert "user-1" not in manager._user_clients

    @pytest.mark.asyncio
    async def test_disconnect_keeps_other_tabs(self, manager, mock_websocket):
        second_websocket = AsyncMock(spec=WebSocket)
        await manager.connect(mock_websocket, "client-1", "user-1")
        await manager.connect(second_websocket, "client-2", "user-1")

        manager.disconnect("client-1")

        assert "client-1" not in manager._user_clients["user-1"]
        assert "client-2" in manager._user_clients["user-1"]

    def test_disconnect_nonexistent_client_is_noop(self, manager):
        manager.disconnect("client-1")
        assert "user-1" not in manager._user_clients

    # ========== send_to_user isolation ==========

    @pytest.mark.asyncio
    async def test_send_to_user_delivers_to_all_own_tabs(self, manager, mock_websocket):
        second_websocket = AsyncMock(spec=WebSocket)
        await manager.connect(mock_websocket, "client-1", "user-1")
        await manager.connect(second_websocket, "client-2", "user-1")

        message = {"type": "notification", "notification": {"id": "n1"}}
        await manager.send_to_user("user-1", message)

        expected = json.dumps(message)
        mock_websocket.send_text.assert_called_once_with(expected)
        second_websocket.send_text.assert_called_once_with(expected)

    @pytest.mark.asyncio
    async def test_send_to_user_does_not_leak_to_other_users(self, manager, mock_websocket):
        other_websocket = AsyncMock(spec=WebSocket)
        await manager.connect(mock_websocket, "client-1", "user-1")
        await manager.connect(other_websocket, "client-2", "user-2")

        await manager.send_to_user("user-1", {"type": "notification"})

        mock_websocket.send_text.assert_called_once()
        other_websocket.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_to_user_no_connections_is_noop(self, manager):
        await manager.send_to_user("user-1", {"type": "notification"})

    @pytest.mark.asyncio
    async def test_send_to_user_disconnects_on_failure(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1", "user-1")
        mock_websocket.send_text.side_effect = Exception("broken pipe")

        await manager.send_to_user("user-1", {"type": "notification"})

        assert "user-1" not in manager._user_clients

    # ========== broadcast ==========

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_every_user(self, manager, mock_websocket):
        other_websocket = AsyncMock(spec=WebSocket)
        await manager.connect(mock_websocket, "client-1", "user-1")
        await manager.connect(other_websocket, "client-2", "user-2")

        await manager.broadcast({"type": "toast"})

        mock_websocket.send_text.assert_called_once()
        other_websocket.send_text.assert_called_once()

    # ========== schedule_send ==========

    @pytest.mark.asyncio
    async def test_schedule_send_from_running_loop_creates_task(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1", "user-1")

        manager.schedule_send("user-1", {"type": "notification"})

        # Give the scheduled task a chance to run.
        await asyncio.sleep(0)

        mock_websocket.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_schedule_send_broadcast_when_user_id_none(self, manager, mock_websocket):
        await manager.connect(mock_websocket, "client-1", "user-1")

        manager.schedule_send(None, {"type": "toast"})
        await asyncio.sleep(0)

        mock_websocket.send_text.assert_called_once()

    def test_schedule_send_outside_loop_uses_captured_loop(self, manager, mock_websocket):
        """
        Simulates a worker-thread caller (e.g. generation completion): no
        running loop in this thread, so schedule_send must bridge via
        run_coroutine_threadsafe against the loop captured by set_loop().
        """
        async def _setup():
            await manager.connect(mock_websocket, "client-1", "user-1")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_setup())
            manager.set_loop(loop)

            # Called from the "main" thread, i.e. with no running loop of
            # its own - get_running_loop() should raise RuntimeError here.
            manager.schedule_send("user-1", {"type": "notification"})

            loop.run_until_complete(asyncio.sleep(0.05))

            mock_websocket.send_text.assert_called_once()
        finally:
            loop.close()

    def test_schedule_send_no_loop_logs_warning_and_does_not_raise(self, manager):
        # No connect()/set_loop() call, so _loop is still None.
        manager.schedule_send("user-1", {"type": "notification"})
