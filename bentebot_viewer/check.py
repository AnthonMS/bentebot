"""Offline native receiver and pixel decoder preflight. Never logs in to Discord."""

import asyncio
import json
import time
from types import SimpleNamespace


def check():
    import av
    import davey
    import discord
    from discord.ext.native_voice import AsyncQueueSink, MediaPacket, StreamClient, VoiceClient
    from discord.stream import StreamKey

    assert discord.__version__.startswith("2.2.")
    assert all((davey, AsyncQueueSink, MediaPacket, StreamClient, VoiceClient))
    assert callable(VoiceClient.watch_stream)
    assert str(StreamKey.from_guild(guild_id=1, channel_id=10, owner_id=100)) == "guild:1:10:100"

    async def sink_contract():
        sink = AsyncQueueSink(
            destination=asyncio.Queue(maxsize=64),
            loop=asyncio.get_running_loop(),
            media_types=["video"],
            drop_oldest=True,
        )
        assert sink.wants_media("video", "H264") and not sink.wants_media("audio", "opus")

    asyncio.run(sink_contract())
    for name in ("h264", "hevc", "vp8", "vp9", "av1"):
        av.CodecContext.create(name, "r")
    from PIL import Image

    from .decode import PixelDecoder

    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height, encoder.pix_fmt = 64, 64, "yuv420p"
    encoder.options = {"preset": "ultrafast", "tune": "zerolatency"}
    frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), "red"))
    encoded = list(encoder.encode(frame)) + list(encoder.encode(None))
    decoder, jpeg = PixelDecoder(), None
    for packet in encoded:
        jpeg = (
            decoder.feed(
                SimpleNamespace(
                    media_type="video", codec="H264", payload=bytes(packet), received_at=time.monotonic()
                )
            )
            or jpeg
        )
    assert jpeg and jpeg.startswith(b"\xff\xd8") and decoder.decoded_frames > 0
    assert decoder.source_size == (64, 64) and decoder.sample_size == (64, 64)
    print(
        json.dumps(
            {"offline_check": "ok", "discord_client": discord.__version__, "encoded_video_to_jpeg": "passed"}
        )
    )


if __name__ == "__main__":
    check()
