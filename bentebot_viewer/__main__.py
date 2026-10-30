from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import time
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import ClientSession, ClientTimeout, web

from .lease import Lease, LeaseState
from .receiver import DiscordReceiver

log = logging.getLogger("bentebot_viewer")


def secret(path):
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        raise ValueError("Viewer secret file is missing or unreadable") from None
    if not 32 <= len(value) <= 256 or not value.isascii():
        raise ValueError("Viewer secret file is empty or invalid")
    return value


async def run():
    import discord

    base = os.getenv("VIEWER_BOT_URL", "http://bentebot:8080").rstrip("/")
    parsed = urlsplit(base)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("VIEWER_BOT_URL must be an HTTP(S) URL without credentials or parameters")
    token = secret(os.getenv("VIEWER_DISCORD_TOKEN_FILE", "/run/secrets/discord_token"))
    key = secret(os.getenv("VIEWER_CONTROL_KEY_FILE", "/run/secrets/viewer_control_key"))
    expected_id = int(os.getenv("VIEWER_USER_ID", "0"))
    if expected_id <= 0 or token.startswith("Bot "):
        raise ValueError("Configure the dedicated viewer User ID and its user-account credential")
    client = discord.Client()
    # Verify the account before opening its gateway. A token from the owner's
    # normal account must never silently turn that account into the viewer.
    try:
        await client.login(token)
        if not client.user or client.user.bot or client.user.id != expected_id:
            raise ValueError("Discord credential does not belong to the configured viewer account")
    except Exception:
        await client.close()
        raise ValueError("Dedicated viewer account verification failed") from None
    log.info("Dedicated viewer account verified; waiting for Bentebot's session authorization")
    state = LeaseState(time.monotonic)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    last_control = 0.0
    sequence = 0
    report_sequence = None
    worker_state = "starting"
    last_error = ""

    async with ClientSession(timeout=ClientTimeout(total=2), trust_env=False) as http:

        async def upload(lease, jpeg):
            nonlocal sequence, report_sequence
            if not state.permits(lease):
                return
            if report_sequence != lease.lease_id:
                sequence, report_sequence = 0, lease.lease_id
            sequence += 1
            async with http.post(
                base + "/screen/frame",
                data=jpeg,
                headers={
                    "Authorization": "Bearer " + lease.frame_token,
                    "Content-Type": "image/jpeg",
                    "X-Bentebot-Sequence": str(sequence),
                },
                allow_redirects=False,
            ) as response:
                if response.status not in {200, 429}:
                    state.current = None
                    raise RuntimeError("Frame upload rejected")

        receiver = DiscordReceiver(client, state, upload, "idle")

        async def control():
            nonlocal last_control, last_error
            while not stop.is_set():
                if not client.is_ready() or not client.user or client.user.bot:
                    state.current = None
                    await asyncio.sleep(0.5)
                    continue
                headers = {"Authorization": "Bearer " + key, "X-Bentebot-Viewer-Id": str(client.user.id)}
                try:
                    started = time.monotonic()
                    async with http.get(
                        base + "/native-viewer/lease", headers=headers, allow_redirects=False
                    ) as response:
                        if response.status != 200:
                            raise RuntimeError(f"bot_rejected_{response.status}")
                        payload = await response.json()
                        state.current = Lease.parse(payload, viewer_id=client.user.id, now=started)
                    last_control, last_error = time.monotonic(), ""
                    current = state.current
                    report = {
                        "state": receiver.status if current else "idle",
                        "decoded_frames": receiver.decoded_frames,
                        "source_size": receiver.source_size,
                        "sample_size": receiver.sample_size,
                    }
                    if current:
                        report["lease_id"] = current.lease_id
                    async with http.post(
                        base + "/native-viewer/report", json=report, headers=headers, allow_redirects=False
                    ) as response:
                        await response.read()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    state.current = None
                    last_error = (
                        str(error)
                        if isinstance(error, RuntimeError) and str(error).startswith("bot_rejected_")
                        else "bot_control_unavailable"
                    )
                await asyncio.sleep(1)

        async def supervise():
            nonlocal worker_state
            task, active, retry_at = None, None, 0.0
            try:
                while not stop.is_set():
                    current = state.current
                    allowed = current is not None and receiver.permitted(current)
                    if task and (not allowed or current != active or task.done()):
                        if task.done() and not task.cancelled() and task.exception():
                            receiver.status = "error"
                            log.warning(
                                "Stream receiver attempt failed (%s; stage=%s)",
                                type(task.exception()).__name__,
                                receiver.stage,
                            )
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        task, active = None, None
                        await receiver.leave()
                        retry_at = time.monotonic() + 3
                    if allowed and task is None and time.monotonic() >= retry_at:
                        active = current
                        task = asyncio.create_task(receiver.watch(current), name="native-stream-receiver")
                        log.info("Starting authorized stream receiver")
                    worker_state = receiver.status if task else "idle"
                    await asyncio.sleep(0.1)
            finally:
                if task:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                await receiver.leave()

        async def health(request):
            ready = client.is_ready() and time.monotonic() - last_control < 10
            return web.json_response(
                {
                    "ready": ready,
                    "state": worker_state,
                    "decoded_frames": receiver.decoded_frames,
                    "error": last_error,
                },
                status=200 if ready else 503,
            )

        app = web.Application()
        app.router.add_get("/health", health)
        server = web.AppRunner(app, access_log=None)
        await server.setup()
        await web.TCPSite(server, "0.0.0.0", 8081).start()
        tasks = [
            asyncio.create_task(client.connect(reconnect=True)),
            asyncio.create_task(control()),
            asyncio.create_task(supervise()),
            asyncio.create_task(stop.wait()),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if not task.cancelled() and task.exception():
                    raise RuntimeError(
                        "Viewer stopped; inspect readiness and account configuration"
                    ) from None
        finally:
            state.current = None
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.close()
            await server.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Isolated Discord native stream viewer")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        from .check import check

        check()
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("discord").setLevel(logging.CRITICAL)
    try:
        asyncio.run(run())
    except (ValueError, RuntimeError):
        log.error(
            "Viewer failed to start or stopped unexpectedly. Check secret files and bot/viewer configuration."
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
