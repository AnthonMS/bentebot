from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bentebot.analysis_inputs import python_intent, text_attachments, validate_python
from bentebot.config import Settings
from bentebot.utility_tools import TOOL_REQUEST, ToolRequest, UtilityTools


def test_analysis_guard_accepts_data_work_and_rejects_obvious_shared_host_changes():
    validate_python(
        "import io, pandas as pd; from IPython.display import display; df=pd.read_csv(io.BytesIO(_inputs['data.csv'])); display(df.sum())"
    )
    for code in [
        "import os; os.system('rm -rf /')",
        "open('/etc/passwd').read()",
        "import requests",
        "import pandas as pd; pd.read_csv('/secret')",
        "().__class__.__base__",
        "exec('print(42)')",
    ]:
        with pytest.raises((ValueError, SyntaxError)):
            validate_python(code)
    assert python_intent("Plot the attached CSV")
    assert not python_intent("Look up the latest release")


async def test_bounded_attachments_are_untrusted_data_not_paths_or_persistent_files():
    readable = AsyncMock(return_value=b"a,b\n1,2\n")
    too_large = AsyncMock()
    inputs, context = await text_attachments(
        SimpleNamespace(
            attachments=[
                SimpleNamespace(filename="../../numbers.csv", size=8, read=readable),
                SimpleNamespace(filename="large.csv", size=200000, read=too_large),
            ]
        )
    )
    assert list(inputs) == [".._.._numbers.csv"]
    assert "Untrusted attachment" in context and "preview may be truncated" in context
    assert "skipped" in context
    too_large.assert_not_awaited()


async def test_python_requires_both_owner_and_trusted_analysis_intent():
    utility = UtilityTools(
        Settings(utility_tools_enabled=True, owner_ids=frozenset({100}), jupyter_token="test")
    )
    utility.jupyter.execute = AsyncMock(return_value={"ok": True, "output": "42"})
    token = TOOL_REQUEST.set(ToolRequest(100, allow_python=False))
    try:
        assert "execute_python" not in {s["function"]["name"] for s in utility.schemas()}
        await utility.execute("execute_python", '{"code":"print(42)"}')
        utility.jupyter.execute.assert_not_awaited()
        TOOL_REQUEST.get().allow_python = True
        assert "execute_python" in {s["function"]["name"] for s in utility.schemas()}
        await utility.execute("execute_python", '{"code":"import os"}')
        utility.jupyter.execute.assert_not_awaited()
        await utility.execute("execute_python", '{"code":"print(42)"}')
        utility.jupyter.execute.assert_awaited_once()
    finally:
        TOOL_REQUEST.reset(token)
        await utility.close()
