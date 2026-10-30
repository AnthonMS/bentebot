"""Check the pinned receive patch against real Opus and a simulated DAVE session.

This exercises ordering/FEC/error paths, not a real Discord MLS handshake.
"""

import struct
from types import SimpleNamespace

import discord
from discord.ext.voice_recv.opus import PacketDecoder
from discord.ext.voice_recv.rtp import FakePacket, RTPPacket


class DaveSession:
    def __init__(self, ready=True):
        self.ready = ready
        self.calls = 0

    def decrypt(self, user_id, media_type, payload):
        self.calls += 1
        if user_id != 100 or not payload.startswith(b"DV"):
            raise ValueError("undecryptable")
        return payload[2:]


def packet(sequence, payload):
    value = RTPPacket(struct.pack(">BBHII", 0x80, 0x78, sequence, sequence * 960, 42) + b"\0")
    value.decrypted_data = payload
    return value


def decoder(*, ready=True, user=100, version=1):
    session = DaveSession(ready)
    client = SimpleNamespace(
        _connection=SimpleNamespace(dave_session=session, dave_protocol_version=version),
        _get_id_from_ssrc=lambda ssrc: user,
        guild=SimpleNamespace(get_member=lambda user_id: None),
        client=SimpleNamespace(get_user=lambda user_id: None),
    )
    sink = SimpleNamespace(voice_client=client, wants_opus=lambda: False)
    router = SimpleNamespace(
        sink=sink, waiter=SimpleNamespace(register=lambda d: None, unregister=lambda d: None)
    )
    return PacketDecoder(router, 42), session


def opus_frame():
    return discord.opus.Encoder().encode(b"\0\1" * 1920, 960)


def test_decrypt_precedes_real_opus_decode():
    value, session = decoder()
    data = value._process_packet(packet(1, b"DV" + opus_frame()))
    assert len(data.pcm) == 3840 and session.calls == 1


def test_unavailable_key_unknown_user_and_bad_payload_drop_safely():
    for kwargs in ({"ready": False}, {"user": None}, {}):
        value, _ = decoder(**kwargs)
        assert value._process_packet(packet(1, b"invalid")) is None
        assert value._last_seq == 1


def test_fec_lookahead_is_decrypted_exactly_once():
    value, session = decoder()
    next_packet = packet(10, b"DV" + opus_frame())
    value._buffer.push(next_packet)
    _, pcm = value._decode_packet(FakePacket(42, 9, 9 * 960))
    assert len(pcm) == 3840
    calls = session.calls
    data = value._process_packet(next_packet)
    assert len(data.pcm) == 3840 and session.calls == calls


def test_silence_plaintext_and_corrupt_opus_do_not_kill_decoder():
    value, session = decoder()
    assert len(value._process_packet(packet(1, b"\xf8\xff\xfe")).pcm) == 3840
    assert session.calls == 0
    value, session = decoder(version=0)
    assert len(value._process_packet(packet(1, b"\xff" * 7)).pcm) == 3840
    assert session.calls == 0
