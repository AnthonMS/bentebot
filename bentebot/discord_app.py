from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import re
import time
from contextvars import ContextVar
from dataclasses import asdict, replace
from typing import Literal

import discord
from aiohttp import web
from discord import app_commands
from discord.ext import commands

from . import __version__
from .analysis_inputs import python_intent, text_attachments
from .code_output import render_discord_output
from .config import Settings
from .consent import VoiceConsent
from .control_feedback import control_acknowledgement
from .conversation import TextConversation, VoiceConversation
from .domain import Audience, Event, Source
from .enrollment import VoiceSpeechBank
from .memory import Memory
from .model_aliases import canonical_model
from .model_warmth import VoiceModelWarmth
from .native_stream import NativeStreamProbe, VoiceMediaAudit
from .native_viewer import NativeViewerBroker, ViewerTarget, control_key
from .permissions import AccessPolicy
from .persona_commands import install_persona
from .providers import OpenAIChat, OpenAITTS, ProviderError, StreamingSTT
from .screen import ScreenBinding, ScreenService
from .storage import Store
from .summaries import SummaryService
from .text import discord_chunks
from .tool_commands import install_tools
from .tts_warmth import VoiceTTSWarmth
from .vision import ScreenAwareness
from .voice import DiscordPlayback, VoiceReceiver
from .voice_control import command_body, semantic_control_candidate, spoken_control
from .voice_profiles import CUSTOM_VOICE, KittenVoiceProfiles

log = logging.getLogger(__name__)


async def ephemeral_status(interaction, text: str) -> None:
    """Requested diagnostics can exceed Discord's per-message limit without losing evidence."""
    for index, chunk in enumerate(discord_chunks(text)):
        deferred = getattr(interaction.response, "is_done", lambda: False)()
        send = interaction.followup.send if index or deferred else interaction.response.send_message
        await send(chunk, ephemeral=True)


def source_for(channel, author_id: int | None = None) -> Source:
    if getattr(channel, "guild", None) is not None:
        return Source("channel", channel.id, channel.guild.id)
    return Source("dm", channel.id, owner_id=author_id)


class DiscordAccess(AccessPolicy):
    def __init__(self, bot: BenteBot):
        self.bot = bot
        super().__init__(self.member_can_read)
        self._threads = ContextVar("permission_threads", default=None)

    @contextlib.asynccontextmanager
    async def snapshot(self):
        token = self._threads.set({}) if self._threads.get() is None else None
        try:
            yield
        finally:
            if token is not None:
                self._threads.reset(token)

    async def allows_all(self, sources, audience):
        async with self.snapshot():
            return await super().allows_all(sources, audience)

    async def member_can_read(self, user_id: int, source: Source) -> bool:
        return await self.member_can_access(user_id, source, history=True)

    async def member_can_access(self, user_id: int, source: Source, *, history: bool) -> bool:
        guild = self.bot.get_guild(source.guild_id)
        if guild is None or guild.id not in self.bot.settings.guild_ids:
            return False
        channel = guild.get_channel_or_thread(source.channel_id)
        if channel is None:
            return False
        member = guild.get_member(user_id)
        if member is None:
            return False
        perms = channel.permissions_for(member)
        if not perms.view_channel or (history and not perms.read_message_history):
            return False
        if isinstance(channel, discord.Thread) and channel.is_private() and not perms.manage_threads:
            try:
                cache = self._threads.get()
                if cache is None or channel.id not in cache:
                    ids = frozenset(item.id for item in await channel.fetch_members())
                    if cache is not None:
                        cache[channel.id] = ids
                else:
                    ids = cache[channel.id]
                if user_id not in ids:
                    return False
            except (discord.HTTPException, discord.NotFound, discord.Forbidden):
                return False
        return True

    async def allows(self, source: Source, audience: Audience) -> bool:
        async with self.snapshot():
            return await self._allows(source, audience)

    async def _allows(self, source: Source, audience: Audience) -> bool:
        if source.kind in {"channel", "screen"}:
            if not self.bot.user or not await self.member_can_read(self.bot.user.id, source):
                return False
        if audience.destination_id:
            if self.bot.user is None:
                return False
            # Text recipients include live readers without Read Message History, not only historians.
            if source.kind != "dm" and source.guild_id != audience.guild_id:
                return False
            guild = self.bot.get_guild(audience.guild_id)
            if guild is None or not guild.chunked:
                # Even an apparently public source can deny access to one uncached member.
                return False
            if source.kind == "dm":
                # A DM can be used in its owner's private bot channel, but never a
                # wider channel. Independently verify the complete destination audience.
                if audience.user_ids != frozenset({source.owner_id}):
                    return False
                destination = Source("channel", audience.destination_id, audience.guild_id)
                if not await self.member_can_access(self.bot.user.id, destination, history=False):
                    return False
                if not await self.member_can_access(source.owner_id, destination, history=False):
                    return False
                for member in guild.members:
                    if member.id not in {source.owner_id, self.bot.user.id} and await self.member_can_access(
                        member.id, destination, history=False
                    ):
                        return False
                return True
        return await super().allows(source, audience)


class BenteBot(commands.Bot):
    def __init__(self, settings: Settings, store: Store | None = None):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(
            command_prefix=commands.when_mentioned,
            intents=intents,
            allowed_mentions=discord.AllowedMentions.none(),
            enable_debug_events=settings.native_stream_probe_enabled,
        )
        self.settings = settings
        self.native_stream_probe = NativeStreamProbe(self)
        self.native_media_audit = VoiceMediaAudit()
        self.native_viewer = None
        self.native_paused_owners: set[tuple[int, int]] = set()
        self.store = store or Store(settings.database_path)
        self.chat = OpenAIChat(settings)
        self.model_warmth = VoiceModelWarmth(settings, self.chat)
        self.tts = OpenAITTS(settings)
        self.tts_warmth = VoiceTTSWarmth(settings, self.tts)
        self.stt = StreamingSTT(settings)
        self.policy = DiscordAccess(self)
        self.memory = Memory(
            self.store,
            self.policy,
            settings.context_chars,
            self.source_label,
            voice_max_chars=settings.voice_context_chars,
        )
        self.text_conversation = TextConversation(self.store, self.memory, self.chat)
        self.summaries = SummaryService(self.store, self.memory, self.chat)
        self.enrollments = VoiceSpeechBank(expiry_seconds=settings.voice_clone_wait_seconds)
        self.voice_jobs: set[asyncio.Task] = set()
        self.profiles = (
            KittenVoiceProfiles(settings.tts_base_url, settings.tts_api_key)
            if settings.voice_profiles_enabled
            else None
        )
        self.voice_options = []
        self.screens = ScreenService(self._screen_binding_valid, frame_ttl=settings.screen_frame_ttl)
        self.vision_chat = OpenAIChat(
            replace(
                settings,
                chat_max_tokens=256,
                max_response_chars=2400,
                generation_timeout=settings.vision_timeout,
                web_tools_enabled=False,
            ),
            client=self.chat.client,
        )
        self.screen_awareness = ScreenAwareness(settings, self.screens, self.policy, self.vision_chat.stream)
        self.voice_sessions: dict[int, tuple[VoiceConversation, VoiceReceiver, DiscordPlayback]] = {}
        self._voice_locks: dict[int, asyncio.Lock] = {}
        self._alone_tasks: dict[int, asyncio.Task] = {}
        self._voice_membership_versions: dict[int, int] = {}
        self.text_tasks: set[asyncio.Task] = set()
        self.background: asyncio.Task | None = None
        self.health: web.AppRunner | None = None
        self.accepting = False
        self.command_sync_names = []
        self._closing = False
        self.tree.interaction_check = self.interaction_allowed
        self.tree.on_error = self.command_error
        register_commands(self)

    async def setup_hook(self):
        self.store.prune(self.settings.retention_days)
        # One global registration serves guilds and allowed DMs. Remove previous guild
        # copies so Discord does not recommend both scopes for the same command.
        registered = await self.tree.sync()
        self.command_sync_names = sorted(command.name for command in registered)
        expected = {command.name for command in self.tree.get_commands()}
        if expected - set(self.command_sync_names):
            raise RuntimeError("Discord did not confirm all current slash commands during startup.")
        log.info("commands_synced version=%s roots=%s", __version__, self.command_sync_names)
        for guild_id in self.settings.guild_ids:
            guild = discord.Object(id=guild_id)
            self.tree.clear_commands(guild=guild)
            await self.tree.sync(guild=guild)
        self.background = asyncio.create_task(self._maintenance(), name="retention-maintenance")
        if self.settings.health_port:
            app = web.Application()
            app.router.add_get("/health/live", self._health_live)
            app.router.add_get("/health/ready", self._health_ready)
            if self.settings.screen_enabled:
                self.screens.install(app)
            if self.settings.native_viewer_enabled:
                self.native_viewer = NativeViewerBroker(
                    self.screens,
                    self._desired_native_view,
                    viewer_id=self.settings.native_viewer_id,
                    key=control_key(self.settings.native_viewer_key_file),
                )
                self.native_viewer.install(app)
            self.health = web.AppRunner(app, access_log=None)
            await self.health.setup()
            await web.TCPSite(self.health, "0.0.0.0", self.settings.health_port).start()

    async def _health_live(self, request):
        return web.json_response(
            {
                "version": __version__,
                "alive": not self._closing,
                "registered_command_roots": self.command_sync_names,
            }
        )

    async def _health_ready(self, request):
        ready = self.is_ready() and self.accepting
        return web.json_response({"ready": ready}, status=200 if ready else 503)

    async def _maintenance(self):
        while True:
            await asyncio.sleep(3600)
            self.store.prune(self.settings.retention_days)

    async def on_ready(self):
        self.accepting = True
        log.info(
            "Bentebot ready version=%s: %s; %s permitted servers",
            __version__,
            self.user.id,
            len(self.settings.guild_ids),
        )

    def source_label(self, source: Source) -> str:
        if source.kind == "dm":
            return "Private DM with its owner"
        guild = self.get_guild(source.guild_id)
        channel = guild.get_channel_or_thread(source.channel_id) if guild else None
        label = f"{guild.name} / #{channel.name}" if channel else str(source.channel_id)
        return (
            f"Screen observation by user {source.owner_id} in {label}" if source.kind == "screen" else label
        )

    async def _screen_binding_valid(self, binding: ScreenBinding) -> bool:
        if not self.settings.screen_enabled:
            return False
        session = self.voice_sessions.get(binding.guild_id)
        if not session or session[0].paused or session[0].source.channel_id != binding.channel_id:
            return False
        guild = self.get_guild(binding.guild_id)
        channel = guild.get_channel(binding.channel_id) if guild else None
        if channel is None:
            return False
        if self.user is None or self.user.id not in {member.id for member in channel.members}:
            return False
        audience = await self.audience_for(channel, voice=True)
        if binding.transport == "native":
            owner = guild.get_member(binding.owner_id)
            if (
                not self.settings.native_viewer_enabled
                or owner is None
                or owner.voice is None
                or not owner.voice.self_stream
                or owner.voice.channel != channel
            ):
                return False
        return (
            audience.user_ids == binding.audience_ids
            and binding.owner_id in audience.user_ids
            and await self.policy.allows(Source("channel", binding.channel_id, binding.guild_id), audience)
        )

    async def _desired_native_view(self):
        if not self.accepting or self._closing or not self.is_ready() or not self.user:
            return None
        for guild_id, (conversation, _, playback) in sorted(self.voice_sessions.items()):
            if guild_id not in self.settings.guild_ids:
                continue
            channel = playback.client.channel
            if (
                channel is None
                or conversation.paused
                or not playback.client.is_connected()
                or channel.id != conversation.source.channel_id
                or self.user.id not in {member.id for member in channel.members}
            ):
                continue
            viewer = channel.guild.get_member(self.settings.native_viewer_id)
            if viewer is None:
                continue
            permissions = channel.permissions_for(viewer)
            if not permissions.view_channel or not permissions.connect:
                continue
            streams = [
                member
                for member in channel.members
                if not member.bot
                and member.id != self.settings.native_viewer_id
                and (guild_id, member.id) not in self.native_paused_owners
                and member.voice
                and member.voice.self_stream
                and member.voice.channel == channel
            ]
            if len(streams) != 1:
                continue
            audience = await self.audience_for(channel, voice=True)
            if not await self.policy.allows(Source("channel", channel.id, guild_id), audience):
                continue
            return ViewerTarget(guild_id, channel.id, streams[0].id, self.user.id, audience.user_ids)
        return None

    async def visual_context(
        self, channel_id: int, audience: Audience, query: str = "", at: float | None = None
    ):
        if not self.settings.screen_enabled or not self.settings.vision_model:
            return [], (), None
        guild = self.get_guild(audience.guild_id)
        channel = guild.get_channel(channel_id) if guild else None
        if channel is None:
            return [], (), None
        call_audience = await self.audience_for(channel, voice=True)
        return await self.screen_awareness.context(channel_id, call_audience, audience, query, at=at)

    async def send_formatted(self, channel, text: str, allowed) -> bool:
        for part in render_discord_output(text):
            if not await allowed():
                return False
            if part.data is not None:
                file = discord.File(io.BytesIO(part.data), filename=part.filename)
                try:
                    await channel.send(
                        part.content or None, file=file, allowed_mentions=discord.AllowedMentions.none()
                    )
                finally:
                    file.close()
            else:
                await channel.send(part.content, allowed_mentions=discord.AllowedMentions.none())
        return True

    def profile_info(self, voice_id: str) -> dict | None:
        value = self.store.get("voice_profiles", voice_id)
        try:
            return json.loads(value) if value else None
        except (ValueError, TypeError):
            return None

    async def begin_enrollment(self, target, name: str, notify, *, consent_basis: str = "individual"):
        if self.profiles is None:
            raise ValueError("Enable VOICE_PROFILES_ENABLED with the official KittenTTS voice API first.")
        current = getattr(getattr(target, "voice", None), "channel", None)
        call = self.voice_sessions.get(target.guild.id)
        if not call or current is None or current.id != call[0].source.channel_id or call[0].paused:
            raise ValueError("The consenting speaker must be in my active voice call.")
        if target.bot or (
            self.settings.native_viewer_enabled and target.id == self.settings.native_viewer_id
        ):
            raise ValueError("Choose a human speaker in this call.")
        capture = self.enrollments.start(
            target.id, current.id, name, duration_seconds=self.settings.voice_clone_speech_seconds
        )
        log.info(
            "voice_clone user=%s channel=%s outcome=collecting target_seconds=%s",
            target.id,
            current.id,
            self.settings.voice_clone_speech_seconds,
        )
        cleaned = False

        def cleanup():
            nonlocal cleaned
            if cleaned:
                return
            cleaned = True
            capture.cancel()

        async def collect():
            sample = b""
            profile = None

            def still_active():
                active = self.voice_sessions.get(target.guild.id)
                voice = getattr(getattr(target, "voice", None), "channel", None)
                return active is call and not call[0].paused and voice is not None and voice.id == current.id

            try:
                sample = await capture.finish(timeout=self.settings.voice_clone_wait_seconds)
                log.info(
                    "voice_clone user=%s channel=%s outcome=reference_ready voiced_seconds=%.1f reference_bytes=%s spliced=%s segments=%s",
                    target.id,
                    current.id,
                    capture.voiced_seconds,
                    len(sample),
                    capture.reference_spliced,
                    capture.reference_segment_count,
                )
                if not still_active():
                    raise ValueError("The voice session changed; the sample was discarded.")
                profile = await self.profiles.create(name, sample, consent_confirmed=True)
                sample = b""
                if not still_active():
                    await self.profiles.delete(profile.id)
                    raise ValueError("The session ended; the newly created profile was removed.")
                try:
                    self.store.set(
                        "voice_profiles",
                        profile.id,
                        json.dumps(
                            {
                                "owner_id": target.id,
                                "guild_id": target.guild.id,
                                "name": profile.name,
                                "consented_at": time.time(),
                                "consent_basis": consent_basis,
                                "reference_spliced": capture.reference_spliced,
                                "reference_segment_count": capture.reference_segment_count,
                            }
                        ),
                    )
                except Exception:
                    await self.profiles.delete(profile.id)
                    raise
                await notify(
                    f"Voice profile saved: `{profile.id}`. Server administrators can select it with "
                    "`/speech voice`. Server administrators can delete it with `/speech delete_voice`. "
                    "The reference is stored by the external TTS service until deletion."
                )
                log.info("voice_clone user=%s channel=%s outcome=saved", target.id, current.id)
            except asyncio.CancelledError:
                log.info("voice_clone user=%s channel=%s outcome=cancelled", target.id, current.id)
                raise
            except (ValueError, TimeoutError, ProviderError) as exc:
                log.warning(
                    "voice_clone user=%s channel=%s outcome=failed error=%s",
                    target.id,
                    current.id,
                    type(exc).__name__,
                )
                await notify(str(exc))
            except Exception:
                log.warning("Voice enrollment failed for user %s", target.id)
                await notify("Voice enrollment failed. Check the external TTS service and storage.")
            finally:
                sample = b""
                cleanup()

        def finished(task):
            self.voice_jobs.discard(task)
            if not task.cancelled() and task.exception() is not None:
                log.warning("Voice enrollment completion notification failed")

        task = asyncio.create_task(collect(), name=f"voice-enrollment-{target.id}")
        self.voice_jobs.add(task)
        task.add_done_callback(finished)
        try:
            await notify(
                f"Voice collection started: {self.settings.voice_clone_speech_seconds} seconds of actual speech, "
                "across pauses. Conversation continues normally; other accounts stay separate. "
                f"Collection expires after {self.settings.voice_clone_wait_seconds // 60} minutes. "
                "Use `/speech cancel_clone` or `/voice pause` to stop collection."
            )
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            cleanup()
            raise

    async def on_disconnect(self):
        self.accepting = False
        self.invalidate()

    async def on_resumed(self):
        # A resumed gateway session dispatches on_resumed rather than another on_ready.
        if not self._closing:
            self.accepting = True

    async def interaction_allowed(self, interaction: discord.Interaction) -> bool:
        allowed = (
            interaction.guild_id in self.settings.guild_ids
            if interaction.guild_id
            else interaction.user.id in self.settings.dm_user_ids
        )
        if not allowed:
            if interaction.type == discord.InteractionType.autocomplete:
                await interaction.response.autocomplete([])
            else:
                await interaction.response.send_message(
                    "This server or DM is not enabled for Bentebot.", ephemeral=True
                )
        return allowed

    async def command_error(self, interaction, error):
        error = getattr(error, "original", error)
        if isinstance(error, (ProviderError, ValueError, app_commands.CheckFailure)):
            message = str(error)
        else:
            log.error(
                "Command failed: %s (%s)",
                interaction.command.qualified_name if interaction.command else "unknown",
                type(error).__name__,
            )
            message = "Command failed. Check bot logs and /diagnostics."
        if interaction.response.is_done():
            await interaction.followup.send(message[:1800], ephemeral=True)
        else:
            await interaction.response.send_message(message[:1800], ephemeral=True)

    def is_admin(self, user) -> bool:
        return user.id in self.settings.owner_ids or (
            isinstance(user, discord.Member) and user.guild_permissions.manage_guild
        )

    def model(self, channel_id: int, guild_id: int | None) -> str:
        return self.model_selection(channel_id, guild_id)[0]

    def model_selection(self, channel_id: int, guild_id: int | None) -> tuple[str, str]:
        if override := self.store.get(f"channel:{channel_id}", "model"):
            return canonical_model(override), "channel override"
        if guild_id is not None and (server := self.store.get(f"guild:{guild_id}", "model")):
            return canonical_model(server), "server default"
        return canonical_model(self.settings.chat_model), "deployment default"

    def model_target(self, interaction, channel=None) -> int:
        if channel is not None:
            return channel.id
        call = self.voice_sessions.get(interaction.guild_id)
        member_channel = getattr(getattr(interaction.user, "voice", None), "channel", None)
        if call and member_channel and member_channel.id == call[0].source.channel_id:
            return member_channel.id
        return interaction.channel_id

    def model_channels(self, interaction):
        guild = interaction.guild
        member = getattr(guild, "me", None) if guild else None
        if not guild or member is None:
            return []
        return [
            channel
            for channel in guild.channels
            if isinstance(channel, (discord.TextChannel, discord.VoiceChannel))
            and channel.permissions_for(interaction.user).view_channel
            and channel.permissions_for(member).view_channel
        ]

    def resolve_model_channel(self, interaction, value: str | None):
        if value is None:
            return None
        choices = self.model_channels(interaction)
        match = re.fullmatch(r"(?:<#)?(\d+)>?", value.strip())
        found = (
            [c for c in choices if c.id == int(match[1])]
            if match
            else [c for c in choices if c.name.casefold() == value.strip().casefold()]
        )
        if len(found) != 1:
            raise ValueError(
                "Choose a text or voice channel both you and the bot can view. Type its name to search, or paste its ID."
            )
        return found[0]

    async def prepare_voice_model(self, guild_id: int, channel_id: int, requested: str) -> None:
        baseline = self.model(channel_id, guild_id)
        self.model_warmth.select(guild_id, baseline)
        # A transient vision override must not replace the session's chat-model hold.
        if requested == baseline:
            await self.model_warmth.prepare(guild_id, baseline)

    async def audience_for(self, channel, *, owner_id: int | None = None, voice: bool = False) -> Audience:
        if getattr(channel, "guild", None) is None:
            return Audience(frozenset({owner_id}))
        guild = channel.guild
        service_ids = {self.user.id}
        if self.settings.native_viewer_enabled:
            service_ids.add(self.settings.native_viewer_id)
        if voice and isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
            ids = frozenset(member.id for member in channel.members if member.id not in service_ids)
            return Audience(ids, guild.id)
        readers = set()
        async with self.policy.snapshot():
            for member in guild.members:
                if member.id in service_ids:
                    continue
                source = Source("channel", channel.id, guild.id)
                if await self.policy.member_can_access(member.id, source, history=False):
                    readers.add(member.id)
        return Audience(frozenset(readers), guild.id, channel.id)

    def invalidate(self, guild_id: int | None = None):
        self.screens.invalidate(guild_id=guild_id)
        self.screen_awareness.clear(guild_id)
        for conversation, _, _ in self.voice_sessions.values():
            if guild_id is None or conversation.source.guild_id == guild_id:
                conversation.invalidate()
        for task in list(self.text_tasks):
            task.cancel()

    async def quiesce(self, guild_id: int | None = None):
        """Drain cancellation finalizers before deleting history they could otherwise reinsert."""
        self.invalidate(guild_id)
        tasks = list(self.text_tasks)
        tasks.extend(
            conversation.task
            for conversation, _, _ in self.voice_sessions.values()
            if conversation.task and (guild_id is None or conversation.source.guild_id == guild_id)
        )
        await asyncio.gather(*tasks, return_exceptions=True)

    async def on_member_update(self, before, after):
        if before.roles != after.roles:
            self.invalidate(after.guild.id)

    async def on_member_join(self, member):
        self.invalidate(member.guild.id)

    async def on_member_remove(self, member):
        self.invalidate(member.guild.id)

    async def on_guild_channel_update(self, before, after):
        self.invalidate(after.guild.id)

    async def on_guild_channel_delete(self, channel):
        self.invalidate(channel.guild.id)

    async def on_guild_role_update(self, before, after):
        self.invalidate(after.guild.id)

    async def on_guild_role_delete(self, role):
        self.invalidate(role.guild.id)

    async def on_thread_update(self, before, after):
        self.invalidate(after.guild.id)

    async def on_thread_delete(self, thread):
        self.invalidate(thread.guild.id)

    async def on_raw_thread_delete(self, payload):
        self.invalidate(payload.guild_id)

    async def on_thread_member_remove(self, member):
        self.invalidate()

    async def on_raw_thread_member_remove(self, payload):
        self.invalidate(payload.guild_id)

    async def on_voice_state_update(self, member, before, after):
        if self.settings.native_viewer_enabled and member.id == self.settings.native_viewer_id:
            return
        if getattr(before, "self_stream", False) and not getattr(after, "self_stream", False):
            self.native_paused_owners.discard((member.guild.id, member.id))
        if before.channel != after.channel:
            self.enrollments.cancel(user_id=member.id)
        if before.channel == after.channel:
            return
        session = self.voice_sessions.get(member.guild.id)
        if session and any(
            channel is not None and channel.id == session[0].source.channel_id
            for channel in (before.channel, after.channel)
        ):
            self._voice_membership_versions[member.guild.id] = (
                self._voice_membership_versions.get(member.guild.id, 0) + 1
            )
            # A listener departure is not a permission revocation or cancellation.
            # Each delivery rechecks source access for the current audience.
            if after.channel is not None and after.channel.id == session[0].source.channel_id:
                if joined := getattr(session[0], "listener_joined", None):
                    joined()
            self.update_alone_timer(member.guild.id)
            active_channel = next(
                c
                for c in (before.channel, after.channel)
                if c is not None and c.id == session[0].source.channel_id
            )
            if setter := getattr(session[1], "set_members", None):
                setter(m.id for m in active_channel.members)
            if before.channel is not None and before.channel.id == session[0].source.channel_id:
                if departed := getattr(session[0], "member_left", None):
                    departed(member.id)
                if drop := getattr(session[1], "drop_speaker", None):
                    await drop(member.id)
        if self.user and member.id == self.user.id:
            session = self.voice_sessions.get(member.guild.id)
            # A move cannot inherit the old channel's source permissions. Stop the entire
            # receiver and require /voice join again instead of relabeling buffered audio.
            # leave_voice removes the session before awaiting, so its own disconnect event
            # cannot recursively disconnect the same client again.
            if session and (after.channel is None or after.channel.id != session[0].source.channel_id):
                await self.leave_voice(member.guild.id)

    async def on_raw_message_delete(self, payload):
        await self.quiesce(payload.guild_id)
        # Conservative removal of all derivatives prevents retained copies of deleted material.
        self.store.forget_channel(payload.channel_id)

    async def on_message_edit(self, before, after):
        # Embed unfurls and bot streaming edits are not changes to stored human text.
        if after.author.bot or before.content == after.content:
            return
        if after.guild:
            if after.guild.id not in self.settings.guild_ids:
                return
            if self.store.get(f"channel:{after.channel.id}", "collect", "true") == "false":
                return
        elif after.author.id not in self.settings.dm_user_ids:
            return
        await self.quiesce(after.guild.id if after.guild else None)
        self.store.forget_channel(after.channel.id)
        await self._remember(after)

    async def _remember(self, message) -> bool:
        if (
            message.author.bot
            or not message.content
            or (self.settings.native_viewer_enabled and message.author.id == self.settings.native_viewer_id)
        ):
            return False
        if message.guild:
            if message.guild.id not in self.settings.guild_ids:
                return False
            if self.store.get(f"channel:{message.channel.id}", "collect", "true") == "false":
                return False
        elif message.author.id not in self.settings.dm_user_ids:
            return False
        source = source_for(message.channel, message.author.id)
        if message.guild and (not self.user or not await self.policy.member_can_read(self.user.id, source)):
            return False
        # The user supplied display name is data, encoded in the prompt rather than interpolated as instructions.
        self.store.save(
            Event(
                source,
                message.author.id,
                message.author.display_name,
                message.content[:16000],
                id=str(message.id),
            )
        )
        return True

    async def send_tool_artifact(self, channel, artifact, allowed):
        if len(artifact.data) > 2_000_000 or not await allowed():
            return False
        await channel.send(
            file=discord.File(io.BytesIO(artifact.data), filename=artifact.name),
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return True

    async def on_message(self, message):
        if message.author.bot:
            return
        mentioned = self.user in message.mentions
        dm = message.guild is None
        if not (self.settings.collect_text or mentioned or dm):
            return
        if not await self._remember(message) or not (mentioned or dm):
            return
        source = source_for(message.channel, message.author.id)

        async def audience():
            return await self.audience_for(message.channel, owner_id=message.author.id)

        async def send(content):
            await message.channel.send(content, allowed_mentions=discord.AllowedMentions.none())

        live_sources: tuple[Source, ...] = ()

        async def deliver(text, origins, recipients):
            async def allowed():
                return (
                    await audience() == recipients
                    and await self.policy.allows_all(origins, recipients)
                    and await self.screens.sources_active(live_sources)
                )

            return await self.send_formatted(message.channel, text, allowed)

        async def artifact_output(artifact, origins, recipients):
            async def allowed():
                return (
                    await audience() == recipients
                    and await self.policy.allows_all(origins, recipients)
                    and await self.screens.sources_active(live_sources)
                )

            return await self.send_tool_artifact(message.channel, artifact, allowed)

        async def visual(recipients):
            nonlocal live_sources
            call = self.voice_sessions.get(source.guild_id)
            if call and message.author.id in (await call[0].audience()).user_ids:
                result = await self.visual_context(call[0].source.channel_id, recipients, message.content)
                live_sources = result[1]
                return result
            live_sources = ()
            return [], (), None

        input_files, attachment_context = await text_attachments(message)
        trigger = Event(
            source,
            message.author.id,
            message.author.display_name,
            message.content[:16000] + ("\n\n" + attachment_context if attachment_context else ""),
            id=str(message.id),
        )

        async def answer():
            try:
                async with message.channel.typing():
                    await self.text_conversation.answer(
                        source,
                        self.user.id,
                        audience,
                        send,
                        self.model(message.channel.id, source.guild_id),
                        trigger=trigger,
                        deliver=deliver,
                        artifact_output=artifact_output,
                        input_files=input_files,
                        python_allowed=python_intent(message.content),
                        extra_context=visual,
                        extra_guard=self.screens.sources_active,
                    )
            except asyncio.CancelledError:
                raise
            except ProviderError as exc:
                await send(str(exc))
            except Exception:
                log.warning("Text response failed in channel %s", message.channel.id)
                with contextlib.suppress(discord.HTTPException):
                    await send("Response failed. Check /diagnostics and the model gateway.")

        # Limit inbound demand; model calls are not allowed to grow without bound.
        if len(self.text_tasks) >= 8:
            await send("I'm handling several conversations. Please try again shortly.")
            return
        task = asyncio.create_task(answer(), name=f"text-answer-{message.channel.id}")
        self.text_tasks.add(task)
        task.add_done_callback(self.text_tasks.discard)

    async def handle_spoken_control(self, guild_id: int, channel_id: int, user_id: int, text: str) -> bool:
        call = self.voice_sessions.get(guild_id)
        output_held = bool(call and getattr(call[0], "output_held", False))
        hold_epoch = getattr(call[0], "output_hold_epoch", 0) if call else 0
        action = spoken_control(text, output_held=output_held)
        log.info(
            "spoken_control_check channel=%s user=%s action=%s words=%s",
            channel_id,
            user_id,
            action or "unmatched",
            len(text.split()),
        )
        if action is None and not semantic_control_candidate(text, output_held=output_held):
            return False
        metrics = getattr(call[0], "metrics", None) if call else None
        control_started = asyncio.get_running_loop().time()
        method = "direct" if action is not None else "intent"
        voice_stamp = getattr(call[0], "speaker_last_voice", {}).get(user_id) if call else None
        revisions = getattr(call[0], "speech_revision", None) if call else None
        speech_revision = revisions.get(user_id, 0) if revisions is not None else None
        control_number = 0
        membership_version = self._voice_membership_versions.get(guild_id, 0)
        if metrics is not None:
            metrics.control_requests += 1
            control_number = metrics.control_requests

        def control_status(result):
            # A slower older job must not overwrite the newest command's status.
            if metrics is not None and metrics.control_requests == control_number:
                metrics.last_control_action = action or "unresolved"
                metrics.last_control_result = result
                metrics.last_control_method = method
                metrics.last_control_ms = (asyncio.get_running_loop().time() - control_started) * 1000

        control_status("checking")
        resumed_epoch: int | None = None

        def authorized(*, check_voice=True):
            guild = self.get_guild(guild_id)
            member = guild.get_member(user_id) if guild else None
            current = getattr(getattr(member, "voice", None), "channel", None)
            return bool(
                not self._closing
                and guild_id in self.settings.guild_ids
                and call is not None
                and self.voice_sessions.get(guild_id) is call
                and call[0].source.channel_id == channel_id
                and member is not None
                and not member.bot
                and not (self.settings.native_viewer_enabled and user_id == self.settings.native_viewer_id)
                and current is not None
                and current.id == channel_id
                and (
                    action not in {"continue", "replace"}
                    or (
                        getattr(call[0], "output_held", False)
                        and getattr(call[0], "output_hold_epoch", 0) == hold_epoch
                        if resumed_epoch is None
                        else not getattr(call[0], "output_held", False)
                        and getattr(call[0], "output_hold_epoch", 0) == resumed_epoch
                    )
                )
                and (
                    not check_voice
                    or (
                        getattr(call[0], "speech_revision", {}).get(user_id, 0) == speech_revision
                        if speech_revision is not None
                        else voice_stamp is None
                        or getattr(call[0], "speaker_last_voice", {}).get(user_id) == voice_stamp
                    )
                )
            )

        def stale_status():
            reason = (
                "newer_transcript"
                if revisions is not None and revisions.get(user_id, 0) != speech_revision
                else "session_membership_or_hold_changed"
            )
            control_status("stale")
            log.info(
                "spoken_control action=%s channel=%s user=%s outcome=stale reason=%s",
                action,
                channel_id,
                user_id,
                reason,
            )

        def leave_after_membership_change():
            # A leave already accepted from a present caller remains an action
            # on this exact call. Membership revokes goodbye audio, not disconnect.
            return bool(
                action == "leave"
                and not self._closing
                and call is not None
                and self.voice_sessions.get(guild_id) is call
                and self._voice_membership_versions.get(guild_id, 0) != membership_version
                and (metrics is None or metrics.control_requests == control_number)
                and (revisions is None or revisions.get(user_id, 0) == speech_revision)
            )

        if not authorized():
            control_status("denied")
            log.info(
                "spoken_control action=%s channel=%s user=%s outcome=denied", action, channel_id, user_id
            )
            return True  # Do not let the model promise an unauthorized action.

        if action is None:
            started = asyncio.get_running_loop().time()
            if output_held:
                action = await self.chat.resolve_spoken_control(
                    text, self.model(channel_id, guild_id), output_held=True
                )
            else:
                action = await self.chat.resolve_spoken_control(text, self.model(channel_id, guild_id))
            log.info(
                "spoken_control_intent channel=%s user=%s action=%s elapsed_ms=%.1f",
                channel_id,
                user_id,
                action or "unmatched",
                (asyncio.get_running_loop().time() - started) * 1000,
            )
            if action == "none":
                control_status("not_control")
                if call:
                    call[0].unresolved_control_request = None
                    call[0].non_control_request = (user_id, text)
                return False
            if action is None:
                control_status("unresolved")
                if call:
                    call[0].unresolved_control_request = (user_id, text)
                return False
            if not authorized():
                stale_status()
                return True

        if action in {"continue", "replace"}:
            if not output_held or not getattr(call[0], "output_held", False):
                control_status("stale")
                return True
            # The person who yielded the bot's turn may hand it back naturally.
            # Other callers should address Bentebot, rather than accidentally
            # resuming it while telling another person to continue.
            if user_id != getattr(call[0], "held_by", None) and not command_body(text)[1]:
                control_status("ignored_side_conversation")
                return True

        control_status("accepted")
        log.info("spoken_control action=%s channel=%s user=%s outcome=accepted", action, channel_id, user_id)

        async def execute():
            nonlocal resumed_epoch
            # A separate task can close the STT task that delivered this request
            # without making that task cancel or await itself.
            if not authorized() and not leave_after_membership_change():
                stale_status()
                return
            conversation = call[0]
            feedback_enabled = bool(getattr(conversation, "control_feedback", None))
            reply_task: asyncio.Task | None = None
            playback_barrier = asyncio.Event()
            original_anchor = getattr(conversation, "held_reply_id", None)

            def start_reply():
                nonlocal resumed_epoch, reply_task
                if not authorized():
                    return
                conversation.release_output(user_id, text, replace=action == "replace")
                resumed_epoch = conversation.output_hold_epoch
                conversation.pending = False
                # The acknowledgement owns playback. Generate into the normal
                # bounded sentence/audio queues until it releases this barrier.
                reply_task = conversation.task = asyncio.create_task(
                    conversation._answer(playback_barrier=playback_barrier), name="voice-answer-overlap"
                )
                reply_task.add_done_callback(lambda _: conversation.changed.set())
                log.info("voice_control_overlap action=%s channel=%s outcome=started", action, channel_id)

            async def perform():
                if action == "cancel_research":
                    await conversation.cancel_research()
                if action == "stop" or feedback_enabled and action in {"leave", "pause"}:
                    await conversation.hold_output(user_id)
                if feedback_enabled:
                    kwargs = {"on_playback": start_reply} if action in {"continue", "replace"} else {}
                    completed = await conversation.acknowledge_control(action, text, authorized, **kwargs)
                    if not completed and not leave_after_membership_change():
                        control_status("completed" if action == "stop" else "stale")
                        return
                # A quick farewell from the caller during our acknowledgement is
                # not a retraction of leave/pause. Explicit barge-in cancels feedback
                # above; session/membership checks still apply before execution.
                if (
                    not authorized(check_voice=not (feedback_enabled and action in {"leave", "pause"}))
                    and not leave_after_membership_change()
                ):
                    control_status("completed" if action == "stop" else "stale")
                    return
                if action == "leave":
                    if metrics is not None and metrics.control_requests != control_number:
                        stale_status()
                        return
                    if feedback_enabled:
                        # Playback completion means packets were sent, not that
                        # Discord's receiver jitter buffer has played the last word.
                        await asyncio.sleep(0.12)
                        if metrics is not None and metrics.control_requests != control_number:
                            stale_status()
                            return
                        if not authorized(check_voice=False) and not leave_after_membership_change():
                            stale_status()
                            return
                    await self.leave_voice(guild_id)
                elif action == "pause":
                    await self.pause_voice(guild_id)
                elif action == "stop":
                    pass  # Output was silenced before the acknowledgement.
                elif action == "cancel_research":
                    pass  # Research was cancelled before the acknowledgement.
                elif action in {"continue", "replace"}:
                    if reply_task is None:
                        conversation.release_output(user_id, text, replace=action == "replace")
                    else:
                        playback_barrier.set()
                elif action == "screen_stop":
                    self.native_paused_owners.add((guild_id, user_id))
                    self.screens.invalidate(owner_id=user_id, guild_id=guild_id)
                    self.screen_awareness.clear(guild_id)
                elif action == "screen_start":
                    self.native_paused_owners.discard((guild_id, user_id))
                control_status("completed")
                log.info(
                    "spoken_control action=%s channel=%s user=%s outcome=completed",
                    action,
                    channel_id,
                    user_id,
                )

            try:
                await perform()
            finally:
                if reply_task is not None and not playback_barrier.is_set():
                    reply_task.cancel()
                    await asyncio.gather(reply_task, return_exceptions=True)
                    if (
                        action == "continue"
                        and not conversation.closed
                        and not conversation.output_held
                        and conversation.output_hold_epoch == resumed_epoch
                        and (conversation.latest_speaker_id, conversation.latest_request) == (user_id, text)
                    ):
                        # An aborted acknowledgement must not give side talk
                        # the turn or discard the original paused answer.
                        conversation.output_held = conversation.metrics.output_held = True
                        conversation.held_by = user_id
                        conversation.held_reply_id = original_anchor
                        conversation.output_hold_epoch += 1
                        conversation.pending = False
                        log.info("voice_output_hold channel=%s outcome=overlap_aborted", channel_id)

        task = asyncio.create_task(execute(), name=f"spoken-control-{action}")
        self.voice_jobs.add(task)

        def finished(done):
            self.voice_jobs.discard(done)
            if not done.cancelled() and done.exception():
                control_status("failed")
                log.warning(
                    "Spoken control failed: action=%s error=%s", action, type(done.exception()).__name__
                )

        task.add_done_callback(finished)
        return True

    async def join_voice(self, member: discord.Member, text_channel) -> str:
        lock = self._voice_locks.setdefault(member.guild.id, asyncio.Lock())
        async with lock:
            if self._closing:
                raise ValueError("Bentebot is shutting down.")
            return await self._join_voice(member, text_channel)

    async def _join_voice(self, member: discord.Member, text_channel) -> str:
        if not self.settings.voice_configured:
            raise ValueError("Voice requires configured streaming STT and TTS APIs. See example.env.")
        if not member.voice or not isinstance(member.voice.channel, discord.VoiceChannel):
            raise ValueError("Join a regular voice channel first (Stage channels are not supported).")
        guild_id = member.guild.id
        if guild_id in self.voice_sessions:
            raise ValueError("I'm already in a voice channel here; use /voice leave first.")
        channel = member.voice.channel
        permissions = channel.permissions_for(member.guild.me)
        if not permissions.connect or not permissions.speak or not permissions.view_channel:
            raise ValueError("I need View Channel, Connect and Speak permissions there.")
        if member.guild.voice_client:
            await member.guild.voice_client.disconnect(force=True)
        client = await channel.connect(cls=voice_recv_client(), self_deaf=False, reconnect=True)
        if client.channel.id != channel.id:
            await client.disconnect(force=True)
            raise ValueError("The voice channel changed while joining; use /voice join again.")
        playback = DiscordPlayback(client)
        source = Source("channel", channel.id, guild_id)

        async def audience():
            return await self.audience_for(client.channel, voice=True)

        async def notify(text):
            with contextlib.suppress(discord.HTTPException):
                await text_channel.send(text)

        async def text_output(text, origins, voice_audience):
            destination = await self.audience_for(text_channel)
            if await self.policy.allows_all(origins, destination):

                async def allowed():
                    current = await self.audience_for(text_channel)
                    return current == destination and await self.policy.allows_all(origins, current)

                return await self.send_formatted(text_channel, text, allowed)
            log.info(
                "voice_text_denied guild=%s channel=%s recipients=%s source_kinds=%s complete_member_cache=%s",
                guild_id,
                text_channel.id,
                len(destination.user_ids),
                sorted({source.kind for source in origins}),
                member.guild.chunked,
            )
            return False

        async def artifact_output(artifact, origins, recipients):
            destination = await self.audience_for(text_channel)

            async def allowed():
                return (
                    await self.audience_for(text_channel) == destination
                    and await self.policy.allows_all(origins, destination)
                    and await self.policy.allows_all(origins, await audience())
                    and not conversation.closed
                )

            return await self.send_tool_artifact(text_channel, artifact, allowed)

        scope = f"guild:{guild_id}"
        conversation = VoiceConversation(
            self.settings,
            self.store,
            self.memory,
            self.chat,
            self.tts,
            source,
            self.user.id,
            audience,
            playback.play,
            playback.stop,
            notify,
            model=lambda: self.model(channel.id, guild_id),
            voice=lambda: self.store.get(scope, "voice", self.settings.tts_voice),
            interruption_words=lambda: int(
                self.store.get(scope, "interruption_words", str(self.settings.interruption_words))
            ),
            text_output=text_output,
            artifact_output=artifact_output,
            text_channel_id=text_channel.id,
            turn_context=lambda recipients, query, at: self.visual_context(channel.id, recipients, query, at),
            extra_guard=self.screens.sources_active,
            text_audience=lambda: self.audience_for(text_channel),
            before_chat=lambda model: self.prepare_voice_model(guild_id, channel.id, model),
            before_tts=lambda voice: self.tts_warmth.prepare(guild_id, voice),
            control=lambda user_id, text: self.handle_spoken_control(guild_id, channel.id, user_id, text),
            control_feedback=lambda action, text: control_acknowledgement(
                self.chat,
                self.model(channel.id, guild_id),
                action,
                text,
                persona=self.memory.personas.prompt(guild_id, channel.id),
            ),
        )
        receiver = VoiceReceiver(client, conversation, self.stt, notify)
        receiver.set_members(m.id for m in channel.members)
        if self.settings.native_viewer_enabled:
            receiver.sink.ignored_user_ids = frozenset({self.settings.native_viewer_id})
        receiver.on_frame = self.enrollments.feed
        # Do not restore a stale single-language preference from an older deployment.
        receiver.language = lambda: "auto"
        self.voice_sessions[guild_id] = (conversation, receiver, playback)
        try:
            conversation.start()
            receiver.start()
            self.model_warmth.select(guild_id, self.model(channel.id, guild_id))
            self.tts_warmth.select(guild_id, conversation.voice())
            self.update_alone_timer(guild_id)
        except Exception:
            await self._leave_voice(guild_id)
            raise
        return f"Joined {channel.mention}."

    async def leave_voice(self, guild_id: int):
        lock = self._voice_locks.setdefault(guild_id, asyncio.Lock())
        async with lock:
            await self._leave_voice(guild_id)

    def voice_has_humans(self, guild_id: int) -> bool | None:
        call = self.voice_sessions.get(guild_id)
        guild = self.get_guild(guild_id)
        if not call or not guild:
            return None
        channel = getattr(guild, "get_channel", lambda _: None)(call[0].source.channel_id)
        if channel is None:
            return None
        return any(
            not member.bot
            and not (self.settings.native_viewer_enabled and member.id == self.settings.native_viewer_id)
            for member in channel.members
        )

    def update_alone_timer(self, guild_id: int):
        occupied = self.voice_has_humans(guild_id)
        call = self.voice_sessions.get(guild_id)
        if call and occupied is not None:
            if presence := getattr(call[0], "set_listener_presence", None):
                presence(occupied)
        if occupied is not False:
            timer = self._alone_tasks.pop(guild_id, None)
            if timer and timer is not asyncio.current_task():
                timer.cancel()
            return
        timer = self._alone_tasks.get(guild_id)
        if timer and not timer.done():
            return
        conversation = self.voice_sessions[guild_id][0]

        async def leave_if_still_alone():
            await asyncio.sleep(self.settings.voice_alone_timeout)
            async with self._voice_locks.setdefault(guild_id, asyncio.Lock()):
                current = self.voice_sessions.get(guild_id)
                if current and current[0] is conversation and self.voice_has_humans(guild_id) is False:
                    log.info(
                        "voice_alone_timeout guild=%s channel=%s", guild_id, conversation.source.channel_id
                    )
                    await self._leave_voice(guild_id)

        timer = self._alone_tasks[guild_id] = asyncio.create_task(
            leave_if_still_alone(), name="voice-alone-leave"
        )

        def finished(done):
            if self._alone_tasks.get(guild_id) is done:
                self._alone_tasks.pop(guild_id, None)
            if not done.cancelled() and done.exception():
                log.warning(
                    "voice_alone_cleanup guild=%s error=%s", guild_id, type(done.exception()).__name__
                )

        timer.add_done_callback(finished)

    async def _leave_voice(self, guild_id: int):
        timer = self._alone_tasks.pop(guild_id, None)
        if timer and timer is not asyncio.current_task():
            timer.cancel()
            await asyncio.gather(timer, return_exceptions=True)
        self.native_paused_owners = {item for item in self.native_paused_owners if item[0] != guild_id}
        self.screens.invalidate(guild_id=guild_id)
        self.screen_awareness.clear(guild_id)
        session = self.voice_sessions.pop(guild_id, None)
        self._voice_membership_versions.pop(guild_id, None)
        cleanup = [self.model_warmth.release(guild_id), self.tts_warmth.release(guild_id)]
        if session:
            conversation, receiver, _ = session
            self.enrollments.cancel(channel_id=conversation.source.channel_id)
            cleanup.extend([conversation.close(), receiver.close()])
        try:
            for result in await asyncio.gather(*cleanup, return_exceptions=True):
                if isinstance(result, BaseException):
                    log.warning("voice_cleanup guild=%s error=%s", guild_id, type(result).__name__)
        finally:
            guild = self.get_guild(guild_id)
            if guild and guild.voice_client:
                await guild.voice_client.disconnect(force=True)

    def _replacement_receiver(self, guild_id: int, conversation, previous):
        receiver = VoiceReceiver(
            previous.client,
            conversation,
            self.stt,
            previous.notify,
            self.enrollments.feed,
            lambda: "auto",
        )
        if self.settings.native_viewer_enabled:
            receiver.sink.ignored_user_ids = frozenset({self.settings.native_viewer_id})
        receiver.sink.allowed_user_ids = getattr(getattr(previous, "sink", None), "allowed_user_ids", None)
        return receiver

    async def pause_voice(self, guild_id: int):
        async with self._voice_locks.setdefault(guild_id, asyncio.Lock()):
            call = self.voice_sessions.get(guild_id)
            if not call:
                raise ValueError("I'm not in a voice call here.")
            conversation, receiver, _ = call
            conversation.paused = True
            if conversation.audio:
                conversation.audio.suspend()
            conversation.pending = False
            self.screens.invalidate(guild_id=guild_id)
            self.screen_awareness.clear(guild_id)
            await conversation.interrupt()
            self.enrollments.cancel(channel_id=conversation.source.channel_id)
            await receiver.close()
            conversation.partial.clear()
            conversation.speaker_started.clear()
            conversation.speaker_clock.clear()
            conversation.speaker_last_voice.clear()

    async def resume_voice(self, guild_id: int, user_id: int | None = None) -> bool:
        async with self._voice_locks.setdefault(guild_id, asyncio.Lock()):
            call = self.voice_sessions.get(guild_id)
            if not call:
                raise ValueError("I'm not in a voice call here.")
            conversation, receiver, playback = call
            if not conversation.paused:
                if conversation.output_held:
                    conversation.release_output(user_id or conversation.held_by, "Please continue")
                    return True
                return False
            replacement = self._replacement_receiver(guild_id, conversation, receiver)
            replacement.start()
            self.voice_sessions[guild_id] = (conversation, replacement, playback)
            conversation.paused = False
            if conversation.output_held:
                conversation.release_output(user_id or conversation.held_by, "Please continue")
            conversation.changed.set()
            return True

    async def close(self):
        if self._closing:
            return
        self._closing = True
        self.accepting = False
        timers = list(self._alone_tasks.values())
        self._alone_tasks.clear()
        for timer in timers:
            timer.cancel()
        await asyncio.gather(*timers, return_exceptions=True)
        if self.native_viewer:
            self.native_viewer.close()
        for guild_id in list(self.voice_sessions):
            await self.leave_voice(guild_id)
        tasks = list(self.text_tasks)
        if self.background:
            tasks.append(self.background)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.health:
            await self.health.cleanup()
        await self.screens.close()
        self.screen_awareness.clear()
        self.enrollments.cancel_all()
        for task in list(self.voice_jobs):
            task.cancel()
        await asyncio.gather(*self.voice_jobs, return_exceptions=True)
        if self.profiles:
            await self.profiles.close()
        await self.model_warmth.close()
        await self.tts_warmth.close()
        await self.chat.close()
        await self.tts.close()
        self.store.close()
        await super().close()


def voice_recv_client():
    from .discord_audio import AudioReceiveClient

    return AudioReceiveClient


def register_commands(bot: BenteBot):
    voice = app_commands.Group(name="voice", description="Join, control and inspect live voice conversations")
    model = app_commands.Group(name="model", description="Choose a model exposed by your gateway")
    speech = app_commands.Group(name="speech", description="Voice and interruption preferences")
    memory = app_commands.Group(name="memory", description="Inspect or remove stored conversation")
    settings = app_commands.Group(name="settings", description="Server and channel preferences")
    screen = app_commands.Group(
        name="screen", description="Inspect and control screen context for the active call"
    )

    def admin(interaction):
        if not bot.is_admin(interaction.user):
            raise app_commands.CheckFailure("Manage Server permission or bot ownership is required.")

    def session(interaction, *, control: bool = True):
        if not interaction.guild_id:
            raise ValueError("Run this command in a server.")
        result = bot.voice_sessions.get(interaction.guild_id)
        if not result:
            raise ValueError("I'm not in a voice call here.")
        if control and not bot.is_admin(interaction.user):
            current = getattr(getattr(interaction.user, "voice", None), "channel", None)
            if current is None or current.id != result[0].source.channel_id:
                raise app_commands.CheckFailure("Join my voice channel to control this session.")
        return result

    @voice.command(name="join", description="Join your voice channel and enable live transcription")
    @app_commands.guild_only()
    async def join(interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        await bot.join_voice(interaction.user, interaction.channel)
        await interaction.delete_original_response()

    @voice.command(name="leave", description="Disconnect and stop all audio processing")
    @app_commands.guild_only()
    async def leave(interaction: discord.Interaction):
        session(interaction)
        await interaction.response.defer(ephemeral=True)
        await bot.leave_voice(interaction.guild_id)
        await interaction.delete_original_response()

    @voice.command(name="stop", description="Interrupt my current response and discard queued speech")
    @app_commands.guild_only()
    async def stop(interaction: discord.Interaction):
        conversation, _, _ = session(interaction)
        await interaction.response.defer(ephemeral=True)
        await conversation.hold_output(interaction.user.id)
        await interaction.followup.send(
            "I'll wait. Ask me to continue or give me a new request.", ephemeral=True
        )

    @voice.command(name="pause", description="Stop listening and speaking until resumed")
    @app_commands.guild_only()
    async def pause(interaction: discord.Interaction):
        session(interaction)
        await interaction.response.defer(ephemeral=True)
        await bot.pause_voice(interaction.guild_id)
        await interaction.followup.send("Listening and speaking paused. Use /voice resume.", ephemeral=True)

    @voice.command(name="resume", description="Resume live transcription and conversation")
    @app_commands.guild_only()
    async def resume(interaction: discord.Interaction):
        session(interaction)
        await interaction.response.defer(ephemeral=True)
        resumed = await bot.resume_voice(interaction.guild_id, interaction.user.id)
        await interaction.followup.send(
            "Conversation resumed." if resumed else "Already listening.", ephemeral=True
        )

    @screen.command(name="inspect", description="Passively count incoming voice media headers for 12 seconds")
    @app_commands.guild_only()
    async def screen_inspect(interaction: discord.Interaction):
        admin(interaction)
        conversation, _, playback = session(interaction)
        requester = getattr(getattr(interaction.user, "voice", None), "channel", None)
        if requester is None or requester.id != conversation.source.channel_id:
            raise ValueError("Join my voice channel before inspecting its media.")
        await interaction.response.defer(ephemeral=True)
        task = asyncio.current_task()
        bot.voice_jobs.add(task)
        try:
            result = await bot.native_media_audit.collect(playback.client)
            await interaction.followup.send(result.describe(), ephemeral=True)
        finally:
            bot.voice_jobs.discard(task)

    @screen.command(name="probe", description="Test experimental native Go Live gateway access")
    @app_commands.guild_only()
    async def screen_probe(interaction: discord.Interaction, streamer: discord.Member | None = None):
        admin(interaction)
        if not bot.settings.native_stream_probe_enabled:
            raise ValueError(
                "Set BENTEBOT_NATIVE_STREAM_PROBE_ENABLED=true and restart to enable this experiment."
            )
        conversation, _, _ = session(interaction)
        target = streamer or interaction.user
        state = getattr(target, "voice", None)
        if (
            target.bot
            or not state
            or not state.channel
            or state.channel.id != conversation.source.channel_id
            or not state.self_stream
        ):
            raise ValueError("Choose a person currently streaming in my voice channel.")
        requester = getattr(getattr(interaction.user, "voice", None), "channel", None)
        if requester is None or requester.id != conversation.source.channel_id:
            raise ValueError("Join my voice channel before testing its stream.")
        await interaction.response.defer(ephemeral=True)
        task = asyncio.current_task()
        bot.voice_jobs.add(task)
        try:
            result = await bot.native_stream_probe.run(
                interaction.guild_id, conversation.source.channel_id, target.id
            )
            await interaction.followup.send(result.describe(), ephemeral=True)
        finally:
            bot.voice_jobs.discard(task)

    @screen.command(name="start", description="Enable your screen context in this voice call")
    @app_commands.guild_only()
    async def screen_start(interaction: discord.Interaction):
        if bot.settings.native_viewer_enabled:
            conversation, _, _ = session(interaction)
            current = getattr(getattr(interaction.user, "voice", None), "channel", None)
            if current is None or current.id != conversation.source.channel_id or conversation.paused:
                raise ValueError("Join my active voice call before enabling your stream.")
            bot.native_paused_owners.discard((interaction.guild_id, interaction.user.id))
            await interaction.response.send_message(
                "Native Discord viewing is enabled for your stream while I am in this call. "
                "Start sharing with Discord's normal Go Live button; no capture helper is needed.",
                ephemeral=True,
            )
            return
        if not bot.settings.screen_enabled or not bot.settings.vision_model:
            raise ValueError(
                "Enable SCREEN_ENABLED and configure a VISION_MODEL alias first. See the README's paired screen capture section."
            )
        conversation, _, _ = session(interaction)
        current = getattr(getattr(interaction.user, "voice", None), "channel", None)
        if current is None or current.id != conversation.source.channel_id or conversation.paused:
            raise ValueError("Join my active voice call before sharing your own screen.")
        recipients = await conversation.audience()
        grant = bot.screens.issue_pair(
            interaction.user.id, interaction.guild_id, current.id, recipients.user_ids
        )
        target = bot.settings.screen_public_url or "http://UNRAID_IP:8768"
        await interaction.response.send_message(
            f"One-time screen pairing code (expires in {grant.expires_in}s): `{grant.code}`\n"
            f"On your PC, run:\n```\npython scripts/screen_companion.py --target {target} --open\n```\n"
            "Paste the code into the local page and explicitly select a window or screen. "
            "Frames are sampled into temporary memory and included only when I answer; no video is recorded. "
            "If call membership or permissions change, pair again. This is a companion capture, not native Go Live reception.",
            ephemeral=True,
        )

    @screen.command(name="stop", description="Stop your screen session and erase buffered frames")
    @app_commands.guild_only()
    async def screen_stop(interaction: discord.Interaction):
        if bot.settings.native_viewer_enabled:
            bot.native_paused_owners.add((interaction.guild_id, interaction.user.id))
        bot.screens.invalidate(guild_id=interaction.guild_id, owner_id=interaction.user.id)
        bot.screen_awareness.clear(interaction.guild_id)
        call = bot.voice_sessions.get(interaction.guild_id)
        if call:
            call[0].invalidate()
        for task in list(bot.text_tasks):
            task.cancel()
        await interaction.response.send_message(
            "Your screen session and buffered frames were removed.", ephemeral=True
        )

    @screen.command(name="status", description="Show your screen pairing and frame freshness")
    @app_commands.guild_only()
    async def screen_status(interaction: discord.Interaction):
        conversation, _, _ = session(interaction, control=False)
        state = bot.screens.describe(interaction.user.id, conversation.source.channel_id)
        if bot.settings.native_viewer_enabled:
            state["native_viewing"] = "enabled"
            state["your_viewing_paused"] = (
                interaction.guild_id,
                interaction.user.id,
            ) in bot.native_paused_owners
            state["vision_model"] = bot.settings.vision_model or "not configured; frame reception only"
        if bot.native_viewer:
            state.update(bot.native_viewer.describe())
        state.update(bot.screen_awareness.describe())
        await interaction.response.send_message(
            "\n".join(f"{key}: {value}" for key, value in state.items()), ephemeral=True
        )

    @voice.command(name="status", description="Show audio stream counters without transcripts")
    @app_commands.guild_only()
    async def status(interaction: discord.Interaction):
        conversation, receiver, _ = session(interaction, control=False)
        await interaction.response.defer(ephemeral=True)
        metrics = asdict(conversation.metrics)
        if research := getattr(conversation, "research", None):
            metrics.update(research.status())
        if presence := getattr(conversation, "listeners_present", None):
            metrics["voice_listeners_present"] = presence.is_set()
            metrics["voice_audio_waiting_for_listeners"] = not presence.is_set()
        if audio := getattr(conversation, "audio", None):
            metrics.update(audio.status())
            metrics.update(audio.speaker_status(interaction.user.id))
        timings = metrics.pop("last_turn")
        chat_metadata = metrics.pop("last_chat")
        selected, selection_scope = bot.model_selection(conversation.source.channel_id, interaction.guild_id)
        metrics.update(voice_model_selected=selected, voice_model_scope=selection_scope)
        chat_metadata.update(await bot.chat.route_metadata(chat_metadata))
        text = "\n".join(
            f"{key}: {round(value, 4 if key.endswith(('_score', '_confidence', '_margin')) else 1) if isinstance(value, float) else value}"
            for key, value in metrics.items()
        )
        if timings:
            text += "\nTurn timings (ms):\n" + "\n".join(
                f"{key}: {value:.1f}" for key, value in timings.items()
            )
        if chat_metadata:
            text += "\nLast/current AI request (may predate a model switch):\n" + "\n".join(
                f"{key}: {value}" for key, value in chat_metadata.items()
            )
        active = sum(bool(speaker.task and not speaker.task.done()) for speaker in receiver.speakers.values())
        queued = sum(speaker.queue.qsize() for speaker in receiver.speakers.values())
        text += "\n" + bot.model_warmth.describe(interaction.guild_id)
        text += "\n" + bot.tts_warmth.describe(interaction.guild_id)
        if conversation.latest_speaker_id == interaction.user.id:
            text += "\nLast recognized speech (you): " + conversation.latest_request[:250]
        pending = conversation.partial.get(interaction.user.id)
        if pending:
            age = time.monotonic() - conversation.partial_updated_at.get(
                interaction.user.id, time.monotonic()
            )
            text += f"\nPending speech (you; provisional, updated {age:.1f}s ago): {pending[:250]}"
        await ephemeral_status(
            interaction,
            f"{'Paused' if conversation.paused else 'Listening'}; {active} active STT streams; {queued * 100} ms queued audio total\n{text}"
            + (f"\n{receiver.last_error}" if receiver.last_error else ""),
        )

    async def model_autocomplete(interaction: discord.Interaction, current: str):
        try:
            models = await bot.chat.models()
        except ProviderError:
            models = [canonical_model(bot.settings.chat_model)]
        return [
            app_commands.Choice(name=name[:100], value=name)
            for name in models
            if current.casefold() in name.casefold() and len(name) <= 100
        ][:25]

    @model.command(name="list", description="List available models and the current selection")
    async def models(interaction: discord.Interaction, channel: str | None = None):
        await interaction.response.defer(ephemeral=True)
        available = await bot.chat.models()
        channel = bot.resolve_model_channel(interaction, channel)
        target_id = bot.model_target(interaction, channel)
        current, origin = bot.model_selection(target_id, interaction.guild_id)
        server = canonical_model(
            bot.store.get(f"guild:{interaction.guild_id}", "model", bot.settings.chat_model)
        )
        call = bot.voice_sessions.get(interaction.guild_id)
        voice_model = bot.model(call[0].source.channel_id, interaction.guild_id) if call else None
        lines = [
            f"Selected for <#{target_id}>: {current} ({origin})",
            f"Server default: {server}" if interaction.guild_id else f"Deployment default: {server}",
        ]
        if call:
            lines.append(f"Active voice <#{call[0].source.channel_id}>: {voice_model}")
        if target_id != interaction.channel_id:
            lines.append(
                f"Command text channel <#{interaction.channel_id}>: {bot.model(interaction.channel_id, interaction.guild_id)}"
            )
        lines.append(r"\* selected for target; [voice] active voice; [server] server default")
        for name in sorted(set(available) | {current, server} | ({voice_model} if voice_model else set())):
            marks = (" [voice]" if name == voice_model else "") + (
                " [server]" if interaction.guild_id and name == server else ""
            )
            if name not in available:
                marks += " [not in gateway list]"
            marker = r"\*" if name == current else " "
            lines.append(f"{marker} {name}{marks}")
        for chunk in discord_chunks("\n".join(lines)):
            await interaction.followup.send(chunk, ephemeral=True)

    @model.command(name="set", description="Select the model for this channel")
    @app_commands.autocomplete(name=model_autocomplete)
    async def set_model(
        interaction: discord.Interaction,
        name: str,
        channel: str | None = None,
        scope: Literal["channel", "server"] = "channel",
    ):
        if interaction.guild:
            admin(interaction)
        await interaction.response.defer(ephemeral=True)
        if name not in await bot.chat.models():
            raise ValueError("Choose a model returned by /model list.")
        channel = bot.resolve_model_channel(interaction, channel)
        if scope == "server" and (channel is not None or interaction.guild_id is None):
            raise ValueError("Server scope requires a server and no channel argument.")
        target_id = bot.model_target(interaction, channel)
        bot.store.set(
            f"guild:{interaction.guild_id}" if scope == "server" else f"channel:{target_id}", "model", name
        )
        call = bot.voice_sessions.get(interaction.guild_id)
        if call and (scope == "server" or call[0].source.channel_id == target_id):
            bot.model_warmth.select(
                interaction.guild_id, bot.model(call[0].source.channel_id, interaction.guild_id)
            )
        message = (
            f"Server default set to {name}; channel overrides take priority."
            if scope == "server"
            else f"Model set to {name} for <#{target_id}>."
        )
        if call:
            effective, origin = bot.model_selection(call[0].source.channel_id, interaction.guild_id)
            message += f"\nActive voice <#{call[0].source.channel_id}>: {effective} ({origin})."
            if (
                getattr(call[0], "task", None)
                and not call[0].task.done()
                or getattr(call[0], "research", None)
                and call[0].research.jobs
            ):
                message += "\nNew requests use the new selection; in-flight requests and research keep their original model."
        await interaction.followup.send(message, ephemeral=True)

    async def model_channel_autocomplete(interaction: discord.Interaction, current: str):
        call = bot.voice_sessions.get(interaction.guild_id)
        voice_id = call[0].source.channel_id if call else None
        query = current.strip().casefold().replace("-", " ")
        matches = [
            c
            for c in bot.model_channels(interaction)
            if query in f"{c.name} {getattr(c.category, 'name', '')} {c.id}".casefold().replace("-", " ")
        ]
        matches.sort(
            key=lambda c: (c.id != voice_id, c.id != interaction.channel_id, c.name.casefold(), c.id)
        )
        return [
            app_commands.Choice(
                name=(
                    f"{'voice' if isinstance(c, discord.VoiceChannel) else 'text'}: {c.name}"
                    + (f" / {c.category.name}" if c.category else "")
                    + (
                        " [active call]"
                        if c.id == voice_id
                        else " [command channel]"
                        if c.id == interaction.channel_id
                        else ""
                    )
                )[:100],
                value=str(c.id),
            )
            for c in matches[:25]
        ]

    models.autocomplete("channel")(model_channel_autocomplete)
    set_model.autocomplete("channel")(model_channel_autocomplete)

    @speech.command(name="voice", description="Choose a voice ID supported by your configured TTS server")
    @app_commands.guild_only()
    async def set_voice(interaction: discord.Interaction, name: str):
        admin(interaction)
        name = name.strip()
        if not name.strip() or len(name) > 100:
            raise ValueError("Voice IDs must contain 1-100 characters.")
        if CUSTOM_VOICE.fullmatch(name):
            info = bot.profile_info(name)
            if not info or info.get("guild_id") != interaction.guild_id:
                raise ValueError("This custom voice has no consenting owner registered in this server.")
        bot.store.set(f"guild:{interaction.guild_id}", "voice", name.strip())
        if interaction.guild_id in bot.voice_sessions:
            bot.tts_warmth.select(interaction.guild_id, name)
        await interaction.response.send_message(f"TTS voice set to {name}.", ephemeral=True)

    @set_voice.autocomplete("name")
    async def voice_autocomplete(interaction: discord.Interaction, current: str):
        names = {
            bot.settings.tts_voice,
            bot.store.get(f"guild:{interaction.guild_id}", "voice"),
            *(name.strip() for name in bot.settings.tts_voices.split(",")),
        }
        for profile in bot.voice_options:
            info = bot.profile_info(profile.id) if profile.custom else None
            if not profile.custom or (info and info.get("guild_id") == interaction.guild_id):
                names.add(profile.id)
        return [
            app_commands.Choice(name=name, value=name)
            for name in sorted(names)
            if name and len(name) <= 100 and current.casefold() in name.casefold()
        ][:25]

    @speech.command(name="voices", description="List voices from the configured KittenTTS service")
    @app_commands.guild_only()
    async def voices(interaction: discord.Interaction):
        if bot.profiles is None:
            raise ValueError("Voice-profile commands require VOICE_PROFILES_ENABLED and KittenTTS.")
        await interaction.response.defer(ephemeral=True)
        bot.voice_options = await bot.profiles.list()
        lines = []
        for profile in bot.voice_options:
            info = bot.profile_info(profile.id) if profile.custom else None
            if profile.custom and (not info or info.get("guild_id") != interaction.guild_id):
                continue
            label = discord.utils.escape_markdown(profile.name.replace("\n", " "))[:100]
            lines.append(f"{label}: `{profile.id}`")
        await interaction.followup.send(
            "\n".join(lines)[:1800] or "No permitted voices found.", ephemeral=True
        )

    @speech.command(name="clone", description="Create a voice profile from a consenting person in this call")
    @app_commands.guild_only()
    async def clone(interaction: discord.Interaction, name: str, user: discord.Member | None = None):
        if bot.profiles is None:
            raise ValueError("Voice cloning requires VOICE_PROFILES_ENABLED and the KittenTTS voice API.")
        conversation, _, _ = session(interaction)
        target = user or interaction.user
        current = getattr(getattr(target, "voice", None), "channel", None)
        if (
            target.bot
            or target.id == bot.settings.native_viewer_id
            or current is None
            or current.id != conversation.source.channel_id
        ):
            raise ValueError("Choose a person in my active voice call.")
        if not 1 <= len(name.strip()) <= 80 or not name.isprintable():
            raise ValueError("Choose a printable name of one to eighty characters.")
        if bot.settings.voice_clone_policy == "server" or target.id == interaction.user.id:
            await interaction.response.defer(ephemeral=True)
            await bot.begin_enrollment(
                target,
                name,
                lambda text: interaction.followup.send(text, ephemeral=True),
                consent_basis="server_policy"
                if bot.settings.voice_clone_policy == "server"
                else "individual",
            )
        else:

            async def allow(consenting):
                await bot.begin_enrollment(
                    target, name, lambda text: consenting.followup.send(text, ephemeral=True)
                )

            await interaction.response.send_message(
                f"{target.mention}, {interaction.user.mention} requests a voice profile. "
                f"Allow collects {bot.settings.voice_clone_speech_seconds} seconds of your actual speech across pauses, "
                "even if others talk over it. Conversation continues while collecting. "
                "The external TTS service retains a reference until a server administrator deletes the profile. "
                "Server administrators may use the saved voice in this server. No collection begins before you allow it.",
                view=VoiceConsent(target.id, allow),
                allowed_mentions=discord.AllowedMentions(users=[target], everyone=False, roles=False),
            )

    async def caller_choices(interaction: discord.Interaction, current: str):
        current = current.lstrip("@").strip()
        call = bot.voice_sessions.get(interaction.guild_id)
        if not call or call[0].paused:
            return []
        channel = interaction.guild.get_channel(call[0].source.channel_id)
        if not channel or interaction.user.id not in {member.id for member in channel.members}:
            return []
        return [
            app_commands.Choice(name=f"{member.display_name} ({member.name})"[:100], value=str(member.id))
            for member in channel.members
            if not member.bot
            and member.id != bot.settings.native_viewer_id
            and current.casefold() in f"{member.display_name} {member.name} {member.id}".casefold()
        ][:25]

    def selected_caller(interaction, value: str | None):
        conversation, _, _ = session(interaction)
        try:
            user_id = (
                int(value.strip().removeprefix("<@").removeprefix("!").removesuffix(">"))
                if value
                else interaction.user.id
            )
        except ValueError as exc:
            raise ValueError("Select a person from the current-call suggestions.") from exc
        target = interaction.guild.get_member(user_id)
        channel = getattr(getattr(target, "voice", None), "channel", None)
        if (
            target is None
            or target.bot
            or target.id == bot.settings.native_viewer_id
            or channel is None
            or channel.id != conversation.source.channel_id
        ):
            raise ValueError("Choose a human speaker who is still in my voice call.")
        return target

    @voice.command(
        name="clone", description="Collect a current caller's voice across pauses, then create a profile"
    )
    @app_commands.guild_only()
    @app_commands.autocomplete(user=caller_choices)
    async def voice_clone(interaction: discord.Interaction, user: str, name: str | None = None):
        target = selected_caller(interaction, user)
        await clone.callback(interaction, name or target.display_name, user=target)

    @voice.command(name="clone_status", description="Show usable speech collected for a current caller")
    @app_commands.guild_only()
    @app_commands.autocomplete(user=caller_choices)
    async def clone_status(interaction: discord.Interaction, user: str | None = None):
        target = selected_caller(interaction, user)
        capture = bot.enrollments.get(target.id)
        text = "No active voice collection for this caller."
        if capture:
            text = "\n".join(f"{key}: {value}" for key, value in capture.stats().items())
        await interaction.response.send_message(text, ephemeral=True)

    @speech.command(name="cancel_clone", description="Cancel your active voice-reference collection")
    @app_commands.guild_only()
    async def cancel_clone(interaction: discord.Interaction):
        count = bot.enrollments.cancel(user_id=interaction.user.id)
        await interaction.response.send_message(
            "Your reference collection was cancelled."
            if count
            else "No reference is being collected. If creation is already underway, wait for its ID and use /speech delete_voice.",
            ephemeral=True,
        )

    @speech.command(
        name="delete_voice", description="Administrator: delete a saved voice profile in this server"
    )
    @app_commands.guild_only()
    async def delete_voice(interaction: discord.Interaction, voice_id: str):
        admin(interaction)
        if bot.profiles is None:
            raise ValueError("The voice-profile API is not configured.")
        info = bot.profile_info(voice_id)
        if not info or info.get("guild_id") != interaction.guild_id:
            raise app_commands.CheckFailure(
                "Administrators can only delete profiles registered in this server."
            )
        await interaction.response.defer(ephemeral=True)
        await bot.quiesce(interaction.guild_id)
        await bot.profiles.delete(voice_id)
        bot.store.remove_setting("voice_profiles", voice_id)
        scope = f"guild:{interaction.guild_id}"
        if bot.store.get(scope, "voice") == voice_id:
            bot.store.remove_setting(scope, "voice")
        await interaction.followup.send(
            "The external service confirmed deletion of the voice profile.", ephemeral=True
        )

    @speech.command(name="interruptions", description="Set how many recognized words interrupt speech")
    @app_commands.guild_only()
    async def interruptions(interaction: discord.Interaction, words: app_commands.Range[int, 2, 20] = 5):
        admin(interaction)
        bot.store.set(f"guild:{interaction.guild_id}", "interruption_words", str(words))
        await interaction.response.send_message(
            f"Interrupt after {words} recognized words; 'stop' works sooner.", ephemeral=True
        )

    @memory.command(name="status", description="Show retention and storage policy")
    async def memory_status(interaction: discord.Interaction):
        await interaction.response.send_message(
            f"Transcript retention: {bot.settings.retention_days} days. No raw audio is stored. "
            "Private context is checked against everyone receiving a reply. /memory forget removes this channel's "
            "stored events and replies derived from it.",
            ephemeral=True,
        )

    @memory.command(name="forget", description="Forget this channel and responses derived from its history")
    async def forget(interaction: discord.Interaction):
        if interaction.guild:
            admin(interaction)
        await interaction.response.defer(ephemeral=True)
        await bot.quiesce(interaction.guild_id)
        count = bot.store.forget_channel(interaction.channel_id)
        await interaction.followup.send(
            f"Removed {count} stored events. Discord messages remain unchanged.", ephemeral=True
        )

    @memory.command(name="search", description="Search older permitted conversation without a model call")
    async def search_memory(interaction: discord.Interaction, query: str):
        await interaction.response.defer(ephemeral=True)
        task = asyncio.current_task()
        if task:
            bot.text_tasks.add(task)
        try:
            recipients = Audience(frozenset({interaction.user.id}), interaction.guild_id)
            results = await bot.memory.search(recipients, query, limit=5)
            lines = []
            for event in results:
                origins = tuple(dict.fromkeys((event.source, *event.provenance)))
                if not await bot.policy.allows_all(origins, recipients):
                    continue
                text = event.content or event.spoken or event.written or event.generated
                label = discord.utils.escape_markdown(bot.source_label(event.source))
                snippet = discord.utils.escape_markdown(text.replace("\n", " ")[:240])
                lines.append(f"**{label}** - {snippet}")
            await interaction.followup.send(
                "\n\n".join(lines)[:1800] or "No permitted matching history found.",
                ephemeral=True,
                allowed_mentions=discord.AllowedMentions.none(),
            )
        finally:
            if task:
                bot.text_tasks.discard(task)

    @memory.command(
        name="summarize", description="Explicitly generate and retain a permission-scoped channel summary"
    )
    async def summarize_memory(interaction: discord.Interaction):
        if interaction.guild:
            admin(interaction)
        await interaction.response.defer(ephemeral=True)
        task = asyncio.current_task()
        if task:
            bot.text_tasks.add(task)
        try:
            recipients = Audience(frozenset({interaction.user.id}), interaction.guild_id)
            source = source_for(interaction.channel, interaction.user.id)
            summary = await bot.summaries.summarize(
                source, recipients, bot.model(interaction.channel_id, interaction.guild_id), bot.user.id
            )
            if not await bot.policy.allows_all((summary.source, *summary.provenance), recipients):
                raise ProviderError("Source access changed; the summary cannot be displayed.")
            await interaction.followup.send(
                summary.content[:1800], ephemeral=True, allowed_mentions=discord.AllowedMentions.none()
            )
        finally:
            if task:
                bot.text_tasks.discard(task)

    @settings.command(name="collection", description="Enable or disable text collection in a channel")
    @app_commands.guild_only()
    async def collection(interaction: discord.Interaction, channel: discord.TextChannel, enabled: bool):
        admin(interaction)
        await interaction.response.defer(ephemeral=True)
        bot.store.set(f"channel:{channel.id}", "collect", str(enabled).lower())
        if not enabled:
            await bot.quiesce(interaction.guild_id)
            bot.store.forget_channel(channel.id)
        await interaction.followup.send(
            f"Text collection {'enabled' if enabled else 'disabled'} in {channel.mention}.", ephemeral=True
        )

    @bot.tree.command(name="diagnostics", description="Check configuration and gateway availability")
    async def diagnostics(interaction: discord.Interaction):
        if interaction.guild:
            admin(interaction)
        await interaction.response.defer(ephemeral=True)
        try:
            count = len(await bot.chat.models())
            chat = f"reachable ({count} models)"
        except ProviderError:
            chat = "unavailable"
        await interaction.followup.send(
            f"Bentebot {__version__}\nChat gateway: {chat}\n"
            f"Voice APIs: {'configured (not yet verified)' if bot.settings.voice_configured else 'not configured'}\n"
            f"Vision: {bot.settings.vision_mode}; {bot.settings.vision_model or 'not configured'}\n"
            f"Web tools: {'configured' if bot.chat.tool_status()['configured'] else 'disabled'}\n"
            f"Voice profiles: {'configured (provider must be verified)' if bot.profiles else 'disabled'}\n"
            "No local inference or paid fallback is built in. Use /voice status for receive/STT counters.",
            ephemeral=True,
        )

    @bot.tree.command(name="help", description="Show Bentebot's commands and conversation behavior")
    async def help_command(interaction: discord.Interaction):
        await interaction.response.send_message(
            "Mention me in enabled text channels, or DM me if allowed. Public server context can cross channels; "
            "private context requires all recipients to have access.\n"
            "`/voice join|leave|stop|pause|resume|status` - live conversation\n"
            "`/model list|set` - gateway models (autocomplete)\n"
            "`/speech voice|voices|interruptions|clone|cancel_clone|delete_voice` - voice preferences and profiles\n"
            "`/memory status|search|summarize|forget` - retrieval, summaries and removal\n"
            "`/persona show|add|edit|remove|reset|undo` - persistent personality rules\n"
            "`/tools status|calculate|convert|time|python` - utilities and owner-only analysis\n"
            "`/screen start|stop|status` - screen context; native Discord viewing when configured\n"
            "`/settings collection` - channel collection\n`/diagnostics` - configuration and gateway status\n"
            "`/voice clone|clone_status` - current-caller cloning when a compatible provider is configured\n"
            "I can use configured read-only web research automatically. Code goes in text; I never edit projects.",
            ephemeral=True,
        )

    install_tools(bot)
    install_persona(bot)
    for group in (voice, model, speech, memory, settings, screen):
        bot.tree.add_command(group)
