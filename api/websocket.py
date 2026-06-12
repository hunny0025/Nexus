"""NEXUS — WebSocket connection manager."""

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time event broadcasting."""

    def __init__(self):
        self.connected_clients: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.connected_clients.append(websocket)
        logger.info(
            f"WebSocket client connected ({len(self.connected_clients)} total)"
        )

    async def disconnect(self, websocket: WebSocket):
        """Remove a disconnected client."""
        async with self._lock:
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)
        logger.info(
            f"WebSocket client disconnected ({len(self.connected_clients)} total)"
        )

    async def broadcast(self, message: dict[str, Any]):
        """Broadcast a JSON message to all connected clients."""
        if not self.connected_clients:
            return

        payload = json.dumps(message, default=str)
        disconnected = []

        async with self._lock:
            clients = list(self.connected_clients)

        for client in clients:
            try:
                await client.send_text(payload)
            except Exception:
                disconnected.append(client)

        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                for client in disconnected:
                    if client in self.connected_clients:
                        self.connected_clients.remove(client)

    @property
    def client_count(self) -> int:
        return len(self.connected_clients)
