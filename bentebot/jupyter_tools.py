"""Bounded jobs on an existing Jupyter server; fresh kernels are not OS sandboxes."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
import websockets
from PIL import Image


@dataclass(frozen=True)
class ToolArtifact:
    name: str
    mime: str
    data: bytes


class JupyterTools:
    def __init__(self, url, token, *, timeout=25, client=None, connector=None):
        self.url = url.rstrip("/")
        self.headers = {"Authorization": "token " + token}
        self.timeout = timeout
        self.client = client or httpx.AsyncClient(timeout=8, follow_redirects=False)
        self.own_client = client is None
        self.connector = connector or websockets.connect
        self.slots = asyncio.Semaphore(2)

    async def _delete(self, kernel):
        # Only an ID returned for this exact job is ever deleted.
        async with asyncio.timeout(8):
            response = await self.client.delete(self.url + "/api/kernels/" + kernel, headers=self.headers)
            if response.status_code not in {204, 404}:
                raise RuntimeError("Owned interpreter kernel cleanup failed.")

    async def execute(self, code, artifacts, allowed, *, inputs=None):
        if not isinstance(code, str) or not code.strip() or len(code) > 12000:
            raise ValueError("Python code must contain 1-12000 characters.")
        inputs = inputs or {}
        if len(inputs) > 2 or sum(len(v) for v in inputs.values()) > 256000:
            raise ValueError("Interpreter inputs exceed the per-job limit.")
        encoded = {name: base64.b64encode(data).decode() for name, data in inputs.items()}
        code = (
            "import base64 as _bb64\n_inputs = {k: _bb64.b64decode(v) for k,v in "
            + repr(encoded)
            + ".items()}\n"
            + code
        )
        kernel = None
        output = []
        size = 0
        failed = False
        initial_artifacts = len(artifacts)
        async with self.slots:
            try:
                async with asyncio.timeout(self.timeout):
                    if not await allowed():
                        raise ValueError("Interpreter request is no longer authorized.")
                    response = await self.client.post(
                        self.url + "/api/kernels", json={"name": "python3"}, headers=self.headers
                    )
                    response.raise_for_status()
                    kernel = str(uuid.UUID(response.json()["id"]))
                    message_id, session_id = uuid.uuid4().hex, uuid.uuid4().hex
                    socket = self.url.replace("https:", "wss:", 1).replace("http:", "ws:", 1)
                    async with self.connector(
                        socket + "/api/kernels/" + kernel + "/channels",
                        additional_headers=self.headers,
                        open_timeout=8,
                        max_size=2_000_000,
                        max_queue=8,
                    ) as ws:
                        if not await allowed():
                            raise ValueError("Interpreter request is no longer authorized.")
                        await ws.send(
                            json.dumps(
                                {
                                    "header": {
                                        "msg_id": message_id,
                                        "session": session_id,
                                        "username": "bentebot",
                                        "msg_type": "execute_request",
                                        "version": "5.3",
                                        "date": datetime.now(UTC).isoformat(),
                                    },
                                    "parent_header": {},
                                    "metadata": {},
                                    "channel": "shell",
                                    "buffers": [],
                                    "content": {
                                        "code": code,
                                        "silent": False,
                                        "store_history": False,
                                        "user_expressions": {},
                                        "allow_stdin": False,
                                        "stop_on_error": True,
                                    },
                                }
                            )
                        )
                        idle = False
                        async for raw in ws:
                            if not isinstance(raw, str):
                                raise ValueError("Unsupported interpreter binary output.")
                            message = json.loads(raw)
                            if message.get("parent_header", {}).get("msg_id") != message_id:
                                continue
                            if not await allowed():
                                raise ValueError("Interpreter request is no longer authorized.")
                            kind = message.get("msg_type") or message.get("header", {}).get("msg_type")
                            content = message.get("content", {})
                            text = ""
                            if kind == "stream":
                                text = content.get("text", "")
                            elif kind == "error":
                                failed = True
                                # Tracebacks can expose paths and source. Return the exception type only.
                                text = "Python error: " + str(content.get("ename", "Error"))[:100]
                            elif kind in {"display_data", "execute_result"}:
                                data = content.get("data", {})
                                text = data.get("text/plain", "")
                                if "image/png" in data and len(artifacts) < 4:
                                    encoded = data["image/png"]
                                    if isinstance(encoded, str) and len(encoded) <= 1_400_000:
                                        image = base64.b64decode(encoded, validate=True)
                                        with Image.open(io.BytesIO(image)) as picture:
                                            if (
                                                picture.format != "PNG"
                                                or picture.width * picture.height > 4_000_000
                                            ):
                                                raise ValueError("Interpreter plot exceeds image limits.")
                                            picture.verify()
                                        if (
                                            not any(
                                                a.mime == "image/png" and a.data == image for a in artifacts
                                            )
                                            and sum(len(a.data) for a in artifacts) + len(image) <= 2_000_000
                                        ):
                                            artifacts.append(
                                                ToolArtifact(
                                                    f"plot-{len(artifacts) + 1}.png", "image/png", image
                                                )
                                            )
                                if "text/csv" in data and len(artifacts) < 4:
                                    csv = data["text/csv"]
                                    if (
                                        isinstance(csv, str)
                                        and len(csv.encode()) <= 200_000
                                        and not any(
                                            a.mime == "text/csv" and a.data == csv.encode() for a in artifacts
                                        )
                                        and sum(len(a.data) for a in artifacts) + len(csv.encode())
                                        <= 2_000_000
                                    ):
                                        artifacts.append(
                                            ToolArtifact(
                                                f"result-{len(artifacts) + 1}.csv", "text/csv", csv.encode()
                                            )
                                        )
                            if isinstance(text, str) and text:
                                size += len(text)
                                if size > 12000:
                                    raise ValueError("Interpreter output exceeded its size limit.")
                                output.append(text)
                            if kind == "status" and content.get("execution_state") == "idle":
                                idle = True
                                break
                        if not idle:
                            raise ValueError("Interpreter closed before execution completed.")
                    if failed:
                        del artifacts[initial_artifacts:]
                    return {
                        "ok": not failed,
                        "output": "\n".join(output),
                        "artifacts": [a.name for a in artifacts],
                    }
            except BaseException:
                del artifacts[initial_artifacts:]
                raise
            finally:
                if kernel:
                    # Drain cleanup even when the foreground response is cancelled.
                    cleanup = asyncio.create_task(self._delete(kernel))
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        await cleanup
                        raise

    async def close(self):
        if self.own_client:
            await self.client.aclose()
