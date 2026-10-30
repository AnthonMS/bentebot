"""Decode continuous encoded frames, sampling a bounded JPEG for the bot."""

from __future__ import annotations

import io
import time


class PixelDecoder:
    def __init__(self):
        self.codec = ""
        self.context = None
        self.last_sample = -100.0
        self.decoded_frames = 0
        self.decode_errors = 0
        self.source_size = None
        self.sample_size = None

    def feed(self, packet, *, now: float | None = None) -> bytes | None:
        import av

        now = time.monotonic() if now is None else now
        names = {"H264": "h264", "H265": "hevc", "VP8": "vp8", "VP9": "vp9", "AV1": "av1"}
        if packet.media_type != "video" or packet.codec.upper() not in names:
            return None
        if not packet.payload or len(packet.payload) > 4_000_000:
            return None
        codec = names[packet.codec.upper()]
        if self.codec != codec:
            self.context = av.CodecContext.create(codec, "r")
            self.context.thread_count = 2
            self.context.options = {"max_pixels": "16000000"}
            self.codec = codec
        result = None
        try:
            frames = self.context.decode(av.Packet(packet.payload))
        except av.error.FFmpegError:
            # A missing keyframe or lost packet should not tear down the call.
            self.decode_errors += 1
            return None
        for frame in frames:
            self.decoded_frames += 1
            self.source_size = (frame.width, frame.height)
            # Decode dependent frames in order, even between JPEG samples. Never
            # relabel a backed-up stream as a fresh image by uploading old frames.
            received = getattr(packet, "received_at", None)
            fresh = received is not None and 0 <= now - received <= 2
            if fresh and now - self.last_sample >= 1 and frame.width * frame.height <= 16_000_000:
                image = frame.to_image()
                image.thumbnail((1600, 1000))
                self.sample_size = image.size
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=82)
                result = output.getvalue()
                self.last_sample = now
        return result
