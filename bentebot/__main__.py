from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from .config import Settings


class VoicePacketNoise(logging.Filter):
    def filter(self, record):
        return not (
            record.levelno <= logging.INFO
            and record.getMessage().startswith("Received unexpected rtcp packet: type=200,")
            and "SenderReportPacket" in record.getMessage()
        )


async def offline_check():
    # Never loads .env, opens a persistent DB, logs in, or calls an inference API.
    from .discord_app import BenteBot
    from .storage import Store

    bot = BenteBot(Settings(database_path=":memory:", health_port=0), Store(":memory:"))
    try:
        names = [command.qualified_name for command in bot.tree.walk_commands()]
        required = {
            "voice join",
            "model set",
            "memory forget",
            "memory search",
            "memory summarize",
            "speech clone",
            "screen start",
            "screen probe",
            "screen inspect",
            "persona show",
            "persona edit",
            "tools python",
            "tools calculate",
            "music play",
            "music stop",
            "music rescan",
            "music download",
            "music ducking",
        }
        if not required.issubset(names):
            raise RuntimeError("Required commands were not registered")

        def validate_options(options):
            for option in options:
                if option.get("max_length", 1) > 6000:
                    raise RuntimeError("Discord string options cannot exceed 6000 characters")
                validate_options(option.get("options", []))

        for command in bot.tree.get_commands():
            validate_options(command.to_dict(bot.tree).get("options", []))
        print(f"Bentebot offline check passed: imports, SQLite, and {len(names)} command entries.")
    finally:
        await bot.close()


async def run_bot(settings: Settings):
    from .discord_app import BenteBot

    bot = BenteBot(settings)
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    with_signal = False
    try:
        loop.add_signal_handler(signal.SIGTERM, stopping.set)
        with_signal = True
    except NotImplementedError:
        pass  # Windows has no Unix signal handlers; asyncio still handles Ctrl+C.
    client = asyncio.create_task(bot.start(settings.discord_token), name="discord-client")
    shutdown = asyncio.create_task(stopping.wait(), name="shutdown-signal")
    try:
        done, _ = await asyncio.wait({client, shutdown}, return_when=asyncio.FIRST_COMPLETED)
        if client in done:
            await client
    finally:
        await bot.close()
        client.cancel()
        shutdown.cancel()
        await asyncio.gather(client, shutdown, return_exceptions=True)
        if with_signal:
            loop.remove_signal_handler(signal.SIGTERM)


def main() -> int:
    parser = argparse.ArgumentParser(description="API-only Discord companion")
    parser.add_argument(
        "--check", action="store_true", help="Offline startup check: no token, network or persistent writes"
    )
    parser.add_argument(
        "--validate-config", action="store_true", help="Validate local configuration without logging in"
    )
    parser.add_argument(
        "--check-commands",
        action="store_true",
        help="Read back Discord registration for the current source; no changes or gateway login",
    )
    args = parser.parse_args()
    if args.check:
        asyncio.run(offline_check())
        return 0
    from dotenv import load_dotenv

    load_dotenv()
    try:
        settings = Settings.from_env()
        settings.validate(login=not args.validate_config)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.validate_config:
        print(
            f"Configuration valid. Voice APIs {'configured' if settings.voice_configured else 'not configured'}."
        )
        return 0
    if args.check_commands:
        from dataclasses import replace

        from .command_registration import verify_registered_commands
        from .discord_app import BenteBot

        async def check_commands():
            bot = BenteBot(replace(settings, database_path=":memory:", health_port=0))
            try:
                expected = {command.qualified_name for command in bot.tree.walk_commands()}
                result = await verify_registered_commands(settings, expected)
                print(
                    f"Discord command registration verified: {result['verified_entries']} entries, including /tools and /persona."
                )
            finally:
                await bot.close()

        try:
            asyncio.run(check_commands())
        except Exception as exc:
            # No provider bodies, credentials or environment dumps in diagnostics.
            print(
                str(exc)
                if isinstance(exc, ValueError)
                else "Could not verify Discord command registration; check connectivity and credentials.",
                file=sys.stderr,
            )
            return 2
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("discord.ext.voice_recv.reader").addFilter(VoicePacketNoise())
    try:
        asyncio.run(run_bot(settings))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
