"""Read back Discord's actual command inventory without editing it or logging in."""

from __future__ import annotations

import httpx


def command_paths(commands, prefix=""):
    paths = set()
    for command in commands:
        if command.get("type", 1) not in {1, 2}:
            continue
        name = (prefix + " " + command["name"]).strip()
        paths.add(name)
        paths.update(command_paths(command.get("options", []), name))
    return paths


async def verify_registered_commands(settings, expected_paths, *, client=None):
    own = client is None
    client = client or httpx.AsyncClient(timeout=10)
    headers = {"Authorization": "Bot " + settings.discord_token}
    try:
        application = await client.get("https://discord.com/api/v10/oauth2/applications/@me", headers=headers)
        application.raise_for_status()
        app_id = application.json()["id"]
        response = await client.get(
            f"https://discord.com/api/v10/applications/{app_id}/commands", headers=headers
        )
        response.raise_for_status()
        actual = command_paths(response.json())
        missing = sorted(set(expected_paths) - actual)
        if missing:
            raise ValueError("Discord has not registered these current commands: " + ", ".join(missing))
        return {"registered_entries": len(actual), "verified_entries": len(expected_paths)}
    finally:
        if own:
            await client.aclose()
