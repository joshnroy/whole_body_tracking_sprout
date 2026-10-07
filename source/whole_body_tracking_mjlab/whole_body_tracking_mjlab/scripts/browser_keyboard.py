"""Small keyboard transport; Viser's command shortcuts don't expose key releases.

Keep the only private Viser API use here: its existing RunJavascriptMessage
installs our client without patching Viser's bundled frontend. Keyboard input
uses a separate JSON WebSocket, not Viser's internal binary wire protocol.
"""

import asyncio
import json
import secrets
import time
from pathlib import Path

from viser import _messages
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


class KeyboardBridge:
    def __init__(self, server, events, host, port):
        self.server = server
        self.events = events
        self.tokens = {}
        self.connections = {}
        self.loop = server.get_event_loop()
        self.source = Path(__file__).with_name("browser_keyboard.js").read_text()

        async def start():
            return await serve(self._handle, host, port, max_size=1024, compression=None)

        self.listener = asyncio.run_coroutine_threadsafe(start(), self.loop).result(timeout=10)

        @server.on_client_connect
        async def connect(client):
            if client.client_id in self.tokens:
                return
            token = secrets.token_urlsafe(32)
            self.tokens[client.client_id] = token
            config = json.dumps({"port": port, "token": token, "client": client.client_id})
            client._websock_connection.queue_message(
                _messages.RunJavascriptMessage(source=f"({self.source})({config});")
            )

        @server.on_client_disconnect
        async def disconnect(client):
            self.tokens.pop(client.client_id, None)
            connection = self.connections.get(client.client_id)
            if connection is not None:
                await connection.close()
            self.events.put(("release", client.client_id))

        async def install_existing():
            # A browser may reconnect while the scene is still being constructed.
            for client in server.get_clients().values():
                await connect(client)

        asyncio.run_coroutine_threadsafe(install_existing(), self.loop).result(timeout=10)

    async def _handle(self, connection):
        client_id = None
        try:
            hello = json.loads(await asyncio.wait_for(connection.recv(), timeout=5))
            if not isinstance(hello, dict):
                return
            candidate = hello.get("client")
            if type(candidate) is not int or self.tokens.get(candidate) != hello.get("token"):
                return
            client_id = candidate
            if client_id in self.connections or client_id not in self.tokens:
                return
            self.connections[client_id] = connection
            await connection.send('{"ready":true}')
            async for raw in connection:
                message = json.loads(raw)
                if not isinstance(message, dict) or client_id not in self.tokens:
                    break
                action = message.get("action")
                if action == "drive":
                    forward, turn = message.get("forward"), message.get("turn")
                    if type(forward) is int and type(turn) is int and forward in (-1, 0, 1) and turn in (-1, 0, 1):
                        self.events.put(("keyboard", (client_id, time.monotonic(), forward, turn)))
                elif action == "release":
                    self.events.put(("release", client_id))
                elif action in ("stop", "toggle_pause", "reset"):
                    self.events.put((action, 0.0))
        except (ConnectionClosed, asyncio.TimeoutError, ValueError, TypeError):
            pass
        finally:
            if client_id is not None and self.connections.get(client_id) is connection:
                self.connections.pop(client_id, None)
                self.events.put(("release", client_id))
            await connection.close()

    def stop(self):
        async def close():
            self.listener.close()
            await self.listener.wait_closed()

        asyncio.run_coroutine_threadsafe(close(), self.loop).result(timeout=10)
