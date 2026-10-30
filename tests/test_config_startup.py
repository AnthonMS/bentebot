import asyncio
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from bentebot.config import Settings
from bentebot.discord_app import BenteBot, source_for


def test_no_secrets_in_settings_representation():
    settings = Settings(
        discord_token="secret1", chat_api_key="secret2", stt_api_key="secret3", tts_api_key="secret4"
    )
    assert "secret" not in repr(settings)


@pytest.mark.parametrize(
    "settings",
    [
        Settings(interruption_words=0),
        Settings(max_speakers=0),
        Settings(retention_days=-1),
        Settings(voice_context_chars=0),
        Settings(turn_silence_ms=10),
        Settings(chat_base_url="file:///etc/passwd"),
        Settings(chat_base_url="http://user:password@host/v1"),
        Settings(stt_url="https://host/asr"),
        Settings(stt_url="wss://paid.example/v1/listen"),
        Settings(tts_base_url="https://paid.example/v1"),
        Settings(stt_url="ws://host/v1/listen?token=secret"),
    ],
)
def test_config_rejects_invalid_and_unapproved_cloud_audio(settings):
    with pytest.raises(ValueError):
        settings.validate(login=False)


def test_config_parsing_and_no_legacy_credential_reuse(monkeypatch):
    for key in list(os.environ):
        if key.startswith("BENTEBOT_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("DISCORD_TOKEN", "legacy-secret")
    monkeypatch.setenv("BENTEBOT_GUILD_IDS", "123, 456")
    monkeypatch.setenv("BENTEBOT_COLLECT_TEXT", "false")
    monkeypatch.setenv("BENTEBOT_VOICE_CONTEXT_CHARS", "12000")
    settings = Settings.from_env()
    assert settings.guild_ids == frozenset({123, 456})
    assert not settings.collect_text and not settings.discord_token
    assert settings.voice_context_chars == 12000
    with pytest.raises(ValueError):
        settings.validate()
    replace(settings, discord_token="new-token").validate()


def test_dm_source_has_no_guild_attribute():
    source = source_for(SimpleNamespace(id=123), 456)
    assert source.owner_id == 456 and source.kind == "dm"


async def test_startup_commands_and_autocomplete_close_cleanly(store):
    bot = BenteBot(Settings(health_port=0), store)
    try:
        assert bot.memory.context_budget(voice=True) == 8000
        assert bot.memory.context_budget() == 24000
        commands = {command.qualified_name: command for command in bot.tree.walk_commands()}
        assert {
            "voice join",
            "voice leave",
            "voice pause",
            "voice resume",
            "voice stop",
            "voice status",
            "model set",
            "speech voice",
            "speech interruptions",
            "memory forget",
            "settings collection",
            "help",
            "diagnostics",
        } <= commands.keys()
        assert commands["model set"]._params["name"].autocomplete
        assert commands["speech voice"]._params["name"].autocomplete
        await asyncio.sleep(0)
    finally:
        # This test owns the store after handing it to the bot.
        await bot.close()


def test_offline_cli_does_not_login_or_create_database(tmp_path):
    root = Path(__file__).parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(root), str(root / ".test-deps")))
    result = subprocess.run(
        [sys.executable, "-m", "bentebot", "--check"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "offline check passed" in result.stdout
    assert list(tmp_path.iterdir()) == []
