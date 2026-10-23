import json
from typing import Dict, List, Optional, Set

from src.platform.websocket.base_connection_hub import BaseConnectionHub

class ConnectionHub(BaseConnectionHub):
    def __init__(self):
        super().__init__()
        self.generation_connections: Dict[str, List[str]] = {}  # generation_id -> [client_ids]
        self._client_generations: Dict[str, Set[str]] = {}
        self.privileged_clients: Set[str] = set()

    def disconnect(self, client_id: str):
        super().disconnect(client_id)
        self.privileged_clients.discard(client_id)

        # Remove from generation connections
        for generation_id in self._client_generations.pop(client_id, ()):
            clients = self.generation_connections.get(generation_id)
            if clients is None:
                continue
            if client_id in clients:
                clients.remove(client_id)
            if not clients:
                del self.generation_connections[generation_id]

    async def subscribe_to_generation(self, client_id: str, generation_id: str, privileged: bool = False):
        # Verify client is in active connections
        if client_id not in self.active_connections:
            return False

        if privileged:
            self.privileged_clients.add(client_id)
        else:
            self.privileged_clients.discard(client_id)

        if generation_id not in self.generation_connections:
            self.generation_connections[generation_id] = []
        if client_id not in self.generation_connections[generation_id]:
            self.generation_connections[generation_id].append(client_id)
        self._client_generations.setdefault(client_id, set()).add(generation_id)
        return True

    async def broadcast_to_generation(self, generation_id: str, message: dict, privileged_message: Optional[dict] = None):
        if generation_id not in self.generation_connections:
            return

        client_ids = self.generation_connections[generation_id].copy()

        privileged_json = None
        if privileged_message is not None:
            try:
                privileged_json = json.dumps(privileged_message)
            except (TypeError, ValueError):
                privileged_json = None

        # First, try to serialize the message to JSON
        try:
            json_message = json.dumps(message)
        except TypeError as e:
            # Try to create a simplified version of the message
            try:
                # Create a simplified version of the message with problematic objects removed
                simplified_message = {
                    'type': message.get('type', 'unknown'),
                    'data': {
                        'error': "Failed to serialize message",
                        'original_type': message.get('type', 'unknown')
                    }
                }
                json_message = json.dumps(simplified_message)
            except Exception:
                return  # Cannot proceed if we can't create a valid JSON message
        except Exception:
            return  # Cannot proceed if we can't create a valid JSON message

        targets = {
            client_id: self.active_connections[client_id]
            for client_id in client_ids
            if client_id in self.active_connections
        }
        await self._send_all(
            targets,
            lambda client_id: privileged_json if privileged_json is not None and client_id in self.privileged_clients else json_message,
            "send generation update",
        )
