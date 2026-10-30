import asyncio
import base64
import io
import json
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from bentebot.config import Settings
from bentebot.jupyter_tools import JupyterTools
from bentebot.utility_tools import TOOL_REQUEST, ToolRequest, UtilityTools, calculate, convert


def test_calculation_and_conversion_are_bounded_and_do_not_eval_python():
    assert calculate("(17 * 23) + 4") == 395
    assert calculate("0.1 + 0.2") == "0.3"
    assert convert(32, "F", "C") == 0
    assert convert(1, "GiB", "MiB") == 1024
    for expression in ["__import__('os').system('whoami')", "2**10000000", "1/0", "True+1", "9e999"]:
        with pytest.raises((ValueError, ArithmeticError)):
            calculate(expression)
    with pytest.raises(ValueError):
        convert(1, "kg", "m")


async def test_requester_authorization_is_application_owned_and_time_uses_real_zone():
    tools = UtilityTools(
        Settings(utility_tools_enabled=True, owner_ids=frozenset({100}), jupyter_token="test")
    )
    assert tools.schemas() == []
    token = TOOL_REQUEST.set(ToolRequest(200))
    try:
        assert "execute_python" not in {s["function"]["name"] for s in tools.schemas()}
        assert "error" in json.loads(await tools.execute("execute_python", '{"code":"print(42)"}'))
        result = json.loads(await tools.execute("current_time", '{"timezone":"Europe/Copenhagen"}'))
        assert result["untrusted_evidence"]["timezone"] == "Europe/Copenhagen"
        assert "error" in json.loads(await tools.execute("calculate", '{"expression":"2+2","requester":100}'))
    finally:
        TOOL_REQUEST.reset(token)
        await tools.close()


@pytest.mark.parametrize("outcome", ["success", "error", "timeout", "revoked", "cancelled"])
async def test_jupyter_deletes_only_owned_kernel_on_all_paths(outcome):
    kernel = str(uuid.uuid4())
    deleted = []
    entered = asyncio.Event()
    picture = io.BytesIO()
    Image.new("RGB", (8, 8)).save(picture, format="PNG")

    def transport(request):
        assert request.headers["Authorization"] == "token private-test"
        if request.method == "POST":
            return httpx.Response(201, json={"id": kernel})
        assert request.method == "DELETE" and request.url.path == "/api/kernels/" + kernel
        deleted.append(kernel)
        return httpx.Response(204)

    class Socket:
        async def send(self, text):
            self.ident = json.loads(text)["header"]["msg_id"]
            entered.set()

        async def __aiter__(self):
            if outcome in {"timeout", "cancelled"}:
                await asyncio.Event().wait()
            for parent, kind, content in [
                ("other-job", "stream", {"text": "unrelated private output"}),
                (self.ident, "stream", {"text": "42"}),
                (
                    self.ident,
                    "error" if outcome == "error" else "display_data",
                    {"ename": "ValueError"}
                    if outcome == "error"
                    else {"data": {"image/png": base64.b64encode(picture.getvalue()).decode()}},
                ),
                (self.ident, "status", {"execution_state": "idle"}),
            ]:
                yield json.dumps(
                    {"header": {"msg_type": kind}, "parent_header": {"msg_id": parent}, "content": content}
                )

    @asynccontextmanager
    async def connect(*args, **kwargs):
        yield Socket()

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        tools = JupyterTools("http://jupyter", "private-test", timeout=0.04, client=client, connector=connect)
        permission = AsyncMock(
            side_effect=[True, True, False] if outcome == "revoked" else None, return_value=True
        )
        artifacts = []
        task = asyncio.create_task(tools.execute("print(42)", artifacts, permission))
        if outcome == "cancelled":
            await entered.wait()
            task.cancel()
        if outcome in {"timeout", "revoked", "cancelled"}:
            with pytest.raises((TimeoutError, ValueError, asyncio.CancelledError)):
                await task
        else:
            result = await task
            assert "unrelated" not in result["output"]
            assert result["ok"] is (outcome == "success")
            assert len(artifacts) == (1 if outcome == "success" else 0)
        assert deleted == [kernel]


async def test_repeated_jupyter_displays_do_not_duplicate_artifacts_or_consume_limits():
    pictures = []
    for color in ("red", "blue"):
        output = io.BytesIO()
        Image.new("RGB", (8, 8), color).save(output, format="PNG")
        pictures.append(output.getvalue())
    kernel = str(uuid.uuid4())
    deleted = []

    def transport(request):
        if request.method == "POST":
            return httpx.Response(201, json={"id": kernel})
        assert request.method == "DELETE" and request.url.path.endswith(kernel)
        deleted.append(kernel)
        return httpx.Response(204)

    class Socket:
        async def send(self, text):
            self.ident = json.loads(text)["header"]["msg_id"]

        async def __aiter__(self):
            displays = [
                ("display_data", {"image/png": base64.b64encode(pictures[0]).decode()}),
                ("execute_result", {"image/png": base64.b64encode(pictures[0]).decode()}),
            ] * 3 + [
                ("display_data", {"image/png": base64.b64encode(pictures[1]).decode()}),
                ("display_data", {"text/csv": "name,value\na,10\n"}),
                ("execute_result", {"text/csv": "name,value\na,10\n"}),
            ]
            for kind, data in displays:
                yield json.dumps(
                    {"msg_type": kind, "parent_header": {"msg_id": self.ident}, "content": {"data": data}}
                )
            yield json.dumps(
                {
                    "msg_type": "status",
                    "parent_header": {"msg_id": self.ident},
                    "content": {"execution_state": "idle"},
                }
            )

    @asynccontextmanager
    async def connect(*args, **kwargs):
        yield Socket()

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        tools = JupyterTools("http://jupyter", "test", client=client, connector=connect)
        artifacts = []
        result = await tools.execute("print(42)", artifacts, AsyncMock(return_value=True))
        assert result["ok"] and len(artifacts) == 3
        assert [a.data for a in artifacts[:2]] == pictures
        assert artifacts[2].mime == "text/csv"
        assert len(set(result["artifacts"])) == 3
        assert deleted == [kernel]
