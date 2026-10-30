from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Source:
    kind: Literal["channel", "dm", "screen"]
    channel_id: int
    guild_id: int | None = None
    owner_id: int | None = None
    session_id: str | None = None
    audience_ids: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "audience_ids", tuple(sorted(set(self.audience_ids))))
        if self.kind not in {"channel", "dm", "screen"}:
            raise ValueError("Unknown source kind")
        if self.kind == "dm" and (not self.owner_id or self.guild_id is not None):
            raise ValueError("DM sources require an owner and no guild")
        if self.kind == "channel" and not self.guild_id:
            raise ValueError("Channel sources require a guild")
        if self.kind == "channel" and self.owner_id is not None:
            raise ValueError("Channel sources cannot have a DM owner")
        if self.kind == "screen" and (
            not self.guild_id or not self.owner_id or not self.session_id or not self.audience_ids
        ):
            raise ValueError("Screen sources require a guild, owner, session and capture audience")
        if self.kind != "screen" and (self.session_id is not None or self.audience_ids):
            raise ValueError("Only screen sources have capture consent metadata")


@dataclass(frozen=True)
class Audience:
    """Humans who can receive a response, or a text destination whose readers must be checked."""

    user_ids: frozenset[int]
    guild_id: int | None = None
    destination_id: int | None = None


@dataclass
class Event:
    source: Source
    author_id: int
    author_name: str
    content: str
    kind: Literal["text", "voice", "assistant", "summary"] = "text"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    # All source permissions are inherited by derived replies and summaries.
    provenance: tuple[Source, ...] = ()
    generated: str = ""
    spoken: str = ""
    interrupted: bool = False
    delivery: Literal["text", "voice", "none"] = "text"
    written: str = ""
    written_channel_id: int | None = None


@dataclass(frozen=True)
class Transcript:
    text: str
    final: bool = False


@dataclass
class ResponseRecord:
    generated: str = ""
    spoken: str = ""
    interrupted: bool = False
    error: str | None = None
    first_audio_ms: float | None = None
    written: str = ""
    written_channel_id: int | None = None
    timings: dict[str, float] = field(default_factory=dict)
    chat_metadata: dict[str, str | int] = field(default_factory=dict)
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    error_stage: str = ""
    web_sources: list[str] = field(default_factory=list)


@dataclass
class Metrics:
    last_turn_id: str = ""
    last_outcome: str = ""
    control_requests: int = 0
    output_held: bool = False
    last_control_action: str = ""
    last_control_result: str = ""
    last_control_method: str = ""
    last_control_ms: float | None = None
    received_frames: int = 0
    dropped_frames: int = 0
    stt_updates: int = 0
    interruptions: int = 0
    provider_errors: int = 0
    responses: int = 0
    stt_interims: int = 0
    stt_finals: int = 0
    stt_first_update_ms: float | None = None
    stt_final_lag_ms: float | None = None
    stt_failures: int = 0
    stt_last_failure_reason: str = ""
    stt_last_close_code: int | None = None
    stt_last_failure_utc: str = ""
    last_turn: dict[str, float] = field(default_factory=dict)
    last_chat: dict[str, str | int] = field(default_factory=dict)
