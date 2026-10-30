"""Native Discord adapter. Imported only inside the isolated viewer image."""

from __future__ import annotations

import asyncio

from .decode import PixelDecoder


class DiscordReceiver:
    def __init__(self, client, state, upload, status):
        self.client, self.state, self.upload, self.status = client, state, upload, status
        self.decoded_frames = 0
        self.stage = "idle"
        self.source_size = None
        self.sample_size = None

    def permitted(self, lease):
        if not self.state.permits(lease) or not self.client.is_ready():
            return False
        guild = self.client.get_guild(lease.guild_id)
        channel = guild.get_channel(lease.channel_id) if guild else None
        bot = guild.get_member(lease.bot_user_id) if guild else None
        owner = guild.get_member(lease.owner_id) if guild else None
        viewer = guild.get_member(lease.viewer_user_id) if guild else None
        if not channel or viewer is None:
            return False
        permissions = channel.permissions_for(viewer)
        return bool(
            permissions.view_channel
            and permissions.connect
            and bot
            and bot.voice
            and bot.voice.channel == channel
            and owner
            and owner.voice
            and owner.voice.channel == channel
            and owner.voice.self_stream
        )

    async def leave(self):
        for voice in list(self.client.voice_clients):
            try:
                async with asyncio.timeout(2):
                    await voice.disconnect(force=True)
            except Exception:
                # Closing the gateway is the final fallback if a voice transport
                # cannot be disconnected; do not leave an unleased watcher behind.
                await self.client.close()
                raise RuntimeError("Viewer transport could not disconnect") from None

    async def watch(self, lease):
        from discord.ext.native_voice import AsyncQueueSink, StreamClient, VoiceClient

        stream_client = None
        self.decoded_frames = 0
        self.source_size = None
        self.sample_size = None
        try:
            if not self.permitted(lease):
                return
            channel = self.client.get_guild(lease.guild_id).get_channel(lease.channel_id)
            # H264 is the initial CPU decoding baseline on Unraid. The receiver
            # still retains support for all codecs if negotiation changes later.
            self.stage = "codec_setup"
            available = await asyncio.to_thread(VoiceClient.get_voice_codecs)
            codecs = [c for c in available if c.type == "audio" or c.name.upper() == "H264"]
            configured = VoiceClient.with_config(codecs=codecs)
            self.status = "connecting"
            if not self.permitted(lease):
                return
            self.stage = "voice_connect"
            voice = await channel.connect(
                cls=configured, self_mute=True, self_deaf=True, self_video=False, timeout=15
            )
            if not self.permitted(lease):
                return
            from discord.stream import StreamKey

            key = StreamKey.from_guild(
                guild_id=lease.guild_id, channel_id=lease.channel_id, owner_id=lease.owner_id
            )
            # STREAM_CREATE can follow the watch request. Waiting for a cached
            # stream before requesting it creates a discovery deadlock.
            self.stage = "stream_connect"
            stream_client = await voice.watch_stream(key, cls=StreamClient, timeout=15, reconnect=False)
            sink = AsyncQueueSink(
                destination=asyncio.Queue(maxsize=64),
                media_types=["video"],
                loop=asyncio.get_running_loop(),
                drop_oldest=True,
            )
            stream_client.listen(sink)
            await stream_client.enable_video_receive()
            await stream_client.ws.media_sink_wants({}, any=100)
            self.status = "watching"
            self.stage = "receiving_video"
            decoder = PixelDecoder()
            while self.permitted(lease) and stream_client.is_connected():
                try:
                    packet = await asyncio.wait_for(sink.queue.get(), timeout=0.5)
                except TimeoutError:
                    continue
                try:
                    # No camera/audio source from a different participant can be
                    # admitted just because it arrived on a shared transport.
                    if packet.user_id != lease.owner_id:
                        continue
                    jpeg = await asyncio.to_thread(decoder.feed, packet)
                    self.decoded_frames = decoder.decoded_frames
                    self.source_size = decoder.source_size
                    self.sample_size = decoder.sample_size
                    if jpeg and self.permitted(lease):
                        self.stage = "frame_upload"
                        await self.upload(lease, jpeg)
                        self.stage = "receiving_video"
                finally:
                    sink.queue.task_done()
        finally:
            if stream_client is not None:
                try:
                    async with asyncio.timeout(2):
                        await stream_client.disconnect(force=True)
                except Exception:
                    pass
            await self.leave()
