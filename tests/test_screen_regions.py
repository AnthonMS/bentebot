import base64
import io

import pytest
from PIL import Image

from bentebot.screen import ScreenSnapshot
from bentebot.screen_regions import crop_frame, region_for
from bentebot.vision import visual_reason


def test_crop_uses_real_pixels_and_preserves_source_and_time():
    image = Image.new("RGB", (100, 80), "red")
    for x in range(50, 100):
        for y in range(40, 80):
            image.putpixel((x, y), (0, 0, 255))
    output = io.BytesIO()
    image.save(output, "PNG")
    frame = ScreenSnapshot(
        100,
        1,
        10,
        frozenset({100}),
        "lease",
        1,
        2,
        "data:image/png;base64," + base64.b64encode(output.getvalue()).decode(),
        100,
        80,
    )
    crop = crop_frame(frame, region_for("look at the bottom right of the screen"))
    assert crop.source == frame.source and crop.received_at == frame.received_at
    assert crop.crop_box == (50, 40, 100, 80) and (crop.width, crop.height) == (50, 40)
    with Image.open(io.BytesIO(base64.b64decode(crop.data_url.split(",")[1]))) as picture:
        assert picture.getpixel((20, 20))[2] > 240
    assert region_for("øverst til venstre") == (0, 0, 0.5, 0.5)
    assert region_for("left and right") is None
    assert region_for("Is this the right answer?") is None
    assert region_for("I left the call") is None
    with pytest.raises(ValueError):
        crop_frame(frame, (1, 1, 0, 0))


@pytest.mark.parametrize(
    "text", ["Why is it doing that?", "There it is again", "Det sker igen", "Hvorfor gør den det?"]
)
def test_implicit_followups_need_recent_visual_focus(text):
    assert visual_reason(text, focused=True) == "followup"
    assert visual_reason(text, focused=False) is None
