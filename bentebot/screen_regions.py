"""Crop received pixels only. A region never widens the original screen grant."""

from __future__ import annotations

import base64
import hashlib
import io
import re
from dataclasses import replace

from PIL import Image


def region_for(query):
    left = bool(re.search(r"\b(left|venstre)\b", query, re.I))
    right = bool(re.search(r"\b(right|højre)\b", query, re.I))
    top = bool(re.search(r"\b(top|upper|øverst|øverste|toppen)\b", query, re.I))
    bottom = bool(re.search(r"\b(bottom|lower|nederst|nederste|bunden)\b", query, re.I))
    if not (top or bottom):
        if not re.search(
            r"\b(?:(?:on|to|at)\s+(?:the\s+)?(?:far\s+)?(?:left|right)|(?:left|right)\s+(?:side|half|corner|part|of)|venstre|højre)\b",
            query,
            re.I,
        ):
            left = right = False
    if left and right or top and bottom or not (left or right or top or bottom):
        return None
    # Half-frame crops preserve actual source resolution; never resize them up.
    return (0.5 if right else 0, 0.5 if bottom else 0, 0.5 if left else 1, 0.5 if top else 1)


def crop_frame(frame, region):
    if frame.width <= 0 or frame.height <= 0:
        raise ValueError("Cannot crop a frame with unknown dimensions.")
    data = base64.b64decode(frame.data_url.split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(data)) as original:
        if original.size != (frame.width, frame.height) or original.width * original.height > 4_000_000:
            raise ValueError("Frame dimensions do not match the received image.")
        box = tuple(
            round(v * (original.width if i % 2 == 0 else original.height)) for i, v in enumerate(region)
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Screen region is empty.")
        output = io.BytesIO()
        cropped = original.crop(box).convert("RGB")
        cropped.save(output, "JPEG", quality=90)
    encoded = output.getvalue()
    return replace(
        frame,
        width=cropped.width,
        height=cropped.height,
        crop_box=box,
        data_url="data:image/jpeg;base64," + base64.b64encode(encoded).decode(),
        fingerprint=hashlib.sha256(encoded).hexdigest(),
    )
