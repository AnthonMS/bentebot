from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bentebot.config import Settings
from bentebot.discord_app import BenteBot
from bentebot.domain import Audience
from bentebot.tool_commands import format_result


@pytest.mark.parametrize(
    "tool,args,evidence,expected",
    [
        ("calculate", {"expression": "0.1 + 0.2"}, {"value": "0.3"}, "0.1 + 0.2 = 0.3"),
        (
            "convert_units",
            {"value": 10, "from_unit": "mi"},
            {"value": 16.09344, "unit": "km"},
            "10 mi = 16.09344 km",
        ),
        (
            "current_time",
            {},
            {"iso_time": "2026-10-30T16:09:11.801668+01:00", "timezone": "Europe/Copenhagen"},
            "2026-10-30 16:09:11 +0100 (Europe/Copenhagen)",
        ),
        ("execute_python", {}, {"ok": True, "output": "42\n"}, "```text\n42\n```"),
    ],
)
def test_explicit_utility_results_are_readable_without_model_inference(tool, args, evidence, expected):
    assert format_result(tool, args, {"untrusted_evidence": evidence}) == expected
    assert "failed" in format_result("execute_python", {}, {"untrusted_evidence": {"ok": False}})
    assert format_result(tool, args, {"error": "Failed", "hint": "Try again"}) == "Failed\nTry again"


async def test_calculate_command_sends_readable_ephemeral_output_after_permission_check(store):
    bot = BenteBot(Settings(health_port=0, utility_tools_enabled=True), store)
    it = SimpleNamespace(
        user=SimpleNamespace(id=100),
        guild_id=1,
        channel_id=10,
        channel=object(),
        response=SimpleNamespace(defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    bot.audience_for = AsyncMock(return_value=Audience(frozenset({100}), 1))
    bot.policy.allows = AsyncMock(return_value=True)
    try:
        await bot.tree.get_command("tools").get_command("calculate").callback(it, "0.1 + 0.2")
        sent = it.followup.send.await_args
        assert sent.args == ("0.1 + 0.2 = 0.3",)
        assert sent.kwargs["ephemeral"] is True
        assert sent.kwargs["allowed_mentions"].everyone is False
        bot.policy.allows.assert_awaited()
    finally:
        await bot.close()
