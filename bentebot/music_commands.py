"""Owner-only music commands, separate from listening and answer controls."""

from __future__ import annotations

import asyncio
import logging
import time

import discord
from discord import app_commands

from .music_intent import GENRES, music_candidate, music_intent
from .music_library import MusicError, normalized

log = logging.getLogger(__name__)


def owner_allowed(bot, guild_id, user_id):
    return bool(
        bot.music
        and guild_id in bot.settings.guild_ids
        and user_id in bot.settings.owner_ids
        and not bot._closing
    )


def authorized(bot, guild_id, user_id):
    if not owner_allowed(bot, guild_id, user_id):
        return False
    guild = bot.get_guild(guild_id)
    call = bot.voice_sessions.get(guild_id)
    member = guild.get_member(user_id) if guild else None
    channel = getattr(getattr(member, "voice", None), "channel", None)
    return bool(
        call
        and member
        and not member.bot
        and user_id != bot.settings.native_viewer_id
        and channel
        and channel.id == call[0].source.channel_id
    )


def session_for(bot, guild_id):
    call = bot.voice_sessions[guild_id]

    async def notify(text):
        # Music metadata comes from the explicitly requested library/catalog,
        # never private conversation history. Stop stale sessions from sending.
        if bot.voice_sessions.get(guild_id) is call:
            await call[1].notify(text)

    return bot.music.session(guild_id, call[2], notify)


async def handle_music_voice(bot, guild_id, channel_id, user_id, text):
    if not getattr(bot, "music", None):
        return False
    intent = music_intent(text)
    existing = bot.music.sessions.get(guild_id) if hasattr(bot.music, "sessions") else None
    context = bool(existing and (existing.tracks or existing.saved))
    if not intent and not music_candidate(text, context=context):
        return False
    if not authorized(bot, guild_id, user_id):
        return bool(intent)
    call = bot.voice_sessions[guild_id]
    if call[0].source.channel_id != channel_id:
        return True
    revision = getattr(call[0], "speech_revision", {}).get(user_id, 0)
    metrics = getattr(call[0], "metrics", None)
    started = time.monotonic()
    if metrics:
        metrics.control_requests += 1
    number = metrics.control_requests if metrics else 0
    method = "direct_music" if intent else "semantic_music"

    def status(result, action="unresolved"):
        if metrics and metrics.control_requests == number:
            metrics.last_control_action = "music_" + action
            metrics.last_control_result = result
            metrics.last_control_method = method
            metrics.last_control_ms = (time.monotonic() - started) * 1000

    def current():
        return (
            authorized(bot, guild_id, user_id)
            and bot.voice_sessions.get(guild_id) is call
            and getattr(call[0], "speech_revision", {}).get(user_id, 0) == revision
        )

    session = session_for(bot, guild_id)
    if not intent:
        status("interpreting")
        intent = await bot.chat.resolve_music_control(
            text,
            bot.model(channel_id, guild_id),
            {
                "active": bool(session.task and not session.task.done()),
                "paused": session.paused,
                "query": session.query,
                "mode": session.mode,
                "retained_playlist": bool(session.saved),
            },
        )
        if not current():
            status("stale")
            return True
        if intent is None:
            status("unresolved")
            await session._notice(
                "I couldn't resolve that music request. Try /music or name the action and music again."
            )
            return True
        if intent[0] == "none":
            status("not_control", "none")
            return False
    action, query, mode = intent
    async with session.commands:
        if not current():
            status("stale", action)
            return True
        try:
            if action == "play":
                await session.start(query, mode)
                await session._notice("Preparing music. The first new recording may take a moment.")
            else:
                await session.control(action)
                await session._notice(f"Music: {action.replace('_', ' ')}.")
            status("accepted", action)
        except MusicError as exc:
            status("failed", action)
            await session._notice(str(exc))
    log.info(
        "music_control method=%s action=%s elapsed_ms=%.1f",
        method,
        action,
        (time.monotonic() - started) * 1000,
    )
    return True


def install_music(bot):
    group = app_commands.Group(
        name="music", description="Owner-only library music, artist playlists and genre radio"
    )

    def checked_owner(interaction):
        if not bot.music:
            raise MusicError("Music is disabled. Enable it and mount the library first.")
        if not owner_allowed(bot, interaction.guild_id, interaction.user.id):
            raise app_commands.CheckFailure("A configured bot owner is required for music maintenance.")

    async def checked(interaction):
        checked_owner(interaction)
        if not authorized(bot, interaction.guild_id, interaction.user.id):
            raise app_commands.CheckFailure(
                "A configured bot owner must be in my voice call to control music."
            )
        return session_for(bot, interaction.guild_id)

    async def suggestions(interaction, current):
        if not owner_allowed(bot, interaction.guild_id, interaction.user.id):
            return []
        indexing = bot.music.start_index()
        if indexing and not indexing.done():
            try:
                await asyncio.wait_for(asyncio.shield(indexing), timeout=0.8)
            except TimeoutError:
                pass
        if not owner_allowed(bot, interaction.guild_id, interaction.user.id):
            return []
        mode = getattr(interaction.namespace, "mode", "auto")
        if mode == "radio":
            names = GENRES | {t.genre for t in bot.music.library.tracks if t.genre}
        elif mode == "track":
            names = {t.artist + " " + t.title for t in bot.music.library.tracks}
        else:
            names = {t.artist for t in bot.music.library.tracks}
        query = normalized(current)
        return [
            app_commands.Choice(name=name, value=name)
            for name in sorted(names, key=str.casefold)
            if name and len(name) <= 100 and query in normalized(name)
        ][:25]

    async def song_suggestions(interaction, current):
        if not owner_allowed(bot, interaction.guild_id, interaction.user.id):
            return []
        artist = normalized(getattr(interaction.namespace, "artist", ""))
        query = normalized(current)
        titles = {t.title for t in bot.music.library.tracks if normalized(t.artist) == artist}
        return [
            app_commands.Choice(name=t, value=t)
            for t in sorted(titles, key=str.casefold)
            if t and len(t) <= 100 and query in normalized(t)
        ][:25]

    @group.command(name="play", description="Play a song, looping artist playlist or continuous genre radio")
    @app_commands.guild_only()
    @app_commands.choices(
        mode=[app_commands.Choice(name=n, value=n) for n in ("auto", "track", "artist", "radio")]
    )
    @app_commands.autocomplete(query=suggestions)
    async def play(
        interaction: discord.Interaction, query: app_commands.Range[str, 1, 180], mode: str = "auto"
    ):
        checked_owner(interaction)
        await interaction.response.defer(ephemeral=True)
        if interaction.guild_id not in bot.voice_sessions:
            await bot.join_voice(interaction.user, interaction.channel)
        session = await checked(interaction)
        async with session.commands:
            await checked(interaction)
            await session.start(query, mode)
        await interaction.followup.send(
            "Preparing music. Use /music status for progress and /music stop to end playback.", ephemeral=True
        )

    async def control(interaction, action):
        session = await checked(interaction)
        await interaction.response.defer(ephemeral=True)
        async with session.commands:
            await checked(interaction)
            await session.control(action)
        await interaction.followup.send(
            session.status(), ephemeral=True, allowed_mentions=discord.AllowedMentions.none()
        )

    @group.command(name="pause", description="Pause music while conversation stays available")
    @app_commands.guild_only()
    async def pause(interaction: discord.Interaction):
        await control(interaction, "pause")

    @group.command(name="resume", description="Resume the paused music")
    @app_commands.guild_only()
    async def resume(interaction: discord.Interaction):
        await control(interaction, "resume")

    @group.command(name="stop", description="Stop music and cancel further preparation")
    @app_commands.guild_only()
    async def stop(interaction: discord.Interaction):
        await control(interaction, "stop")

    @group.command(name="next", description="Skip to the next recording")
    @app_commands.guild_only()
    async def next_track(interaction: discord.Interaction):
        await control(interaction, "next")

    @group.command(name="previous", description="Play the previous recording from its beginning")
    @app_commands.guild_only()
    async def previous(interaction: discord.Interaction):
        await control(interaction, "previous")

    @group.command(name="restart", description="Restart the current recording")
    @app_commands.guild_only()
    async def restart(interaction: discord.Interaction):
        await control(interaction, "restart")

    @group.command(name="status", description="Show current recording, queue and preparation issues")
    @app_commands.guild_only()
    async def status(interaction: discord.Interaction):
        checked_owner(interaction)
        session = bot.music.sessions.get(interaction.guild_id)
        text = (
            session.status()
            if session
            else f"Music: stopped\nLibrary: {len(bot.music.library.tracks)} indexed recordings; maintenance jobs: {len(bot.music.jobs)}"
        )
        await interaction.response.send_message(
            text, ephemeral=True, allowed_mentions=discord.AllowedMentions.none()
        )

    @group.command(name="queue", description="Show up to 15 upcoming recordings")
    @app_commands.guild_only()
    async def queue(interaction: discord.Interaction):
        checked_owner(interaction)
        session = bot.music.sessions.get(interaction.guild_id)
        tracks = session.tracks if session else []
        text = "\n".join(
            f"{i + 1}. {tracks[(session.position + i) % len(tracks)].artist} - {tracks[(session.position + i) % len(tracks)].title}"
            for i in range(min(15, len(tracks)))
        )
        await interaction.response.send_message(
            (text or "No playlist.")[:1900], ephemeral=True, allowed_mentions=discord.AllowedMentions.none()
        )

    @group.command(name="volume", description="Set music volume from zero to 100 percent")
    @app_commands.guild_only()
    async def volume(interaction: discord.Interaction, percent: app_commands.Range[int, 0, 100]):
        session = await checked(interaction)
        session.playback.mixer.volume = percent / 100
        await interaction.response.send_message(f"Music volume: {percent}%", ephemeral=True)

    @group.command(name="rescan", description="Refresh the local library metadata index")
    @app_commands.guild_only()
    async def rescan(interaction: discord.Interaction):
        checked_owner(interaction)
        await interaction.response.defer(ephemeral=True)
        count = await bot.music.rescan()
        await interaction.followup.send(f"Indexed {count} local recordings.", ephemeral=True)

    @group.command(name="ducking", description="Choose whether recognized caller speech lowers music volume")
    @app_commands.guild_only()
    async def ducking(interaction: discord.Interaction, callers: bool):
        session = await checked(interaction)
        session.playback.duck_humans = callers
        if not callers:
            session.playback.speech_gate.users.clear()
        await interaction.response.send_message(
            f"Caller speech ducking: {'on' if callers else 'off'}. Bot speech still lowers music.",
            ephemeral=True,
        )

    @group.command(
        name="download", description="Download one artist/song into the library without joining voice"
    )
    @app_commands.guild_only()
    @app_commands.autocomplete(artist=suggestions, title=song_suggestions)
    async def download(
        interaction: discord.Interaction,
        artist: app_commands.Range[str, 1, 180],
        title: app_commands.Range[str, 1, 180],
    ):
        checked_owner(interaction)
        await interaction.response.defer(ephemeral=True)
        track = await bot.music.download(artist, title)
        await interaction.followup.send(
            f"Available in the library: {track.artist} - {track.title}",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    bot.tree.add_command(group)
