from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit


def _ids(value: str) -> frozenset[int]:
    return frozenset(int(item.strip()) for item in value.split(",") if item.strip())


def _bool(value: str) -> bool:
    if value.lower() not in {"true", "false", "1", "0"}:
        raise ValueError("Boolean settings must be true, false, 1 or 0")
    return value.lower() in {"true", "1"}


@dataclass(frozen=True)
class Settings:
    discord_token: str = field(default="", repr=False)
    guild_ids: frozenset[int] = frozenset()
    owner_ids: frozenset[int] = frozenset()
    dm_user_ids: frozenset[int] = frozenset()
    database_path: str = "data/bentebot.sqlite3"
    chat_base_url: str = "http://host.docker.internal:4000/v1"
    chat_api_key: str = field(default="", repr=False)
    chat_model: str = "bentebot-chat"
    chat_max_tokens: int = 2048
    chat_reasoning_effort: str = ""
    chat_reasoning_models: str = ""
    chat_template_no_thinking_models: str = ""
    voice_warm_models: str = ""
    voice_warm_interval: float = 120.0
    voice_warm_timeout: float = 120.0
    generation_timeout: float = 120.0
    voice_first_content_timeout: float = 15.0
    vision_model: str = ""
    vision_mode: str = "direct"
    vision_timeout: float = 8.0
    vision_requests_per_hour: int = 60
    utility_tools_enabled: bool = False
    jupyter_url: str = "http://host.docker.internal:8888"
    jupyter_token: str = field(default="", repr=False)
    jupyter_timeout: float = 25.0
    web_tools_enabled: bool = False
    web_search_url: str = ""
    web_tool_models: str = ""
    native_web_openai_models: str = ""
    native_web_anthropic_models: str = ""
    web_tool_timeout: float = 10.0
    web_max_tool_rounds: int = 2
    screen_enabled: bool = False
    native_stream_probe_enabled: bool = False
    native_viewer_enabled: bool = False
    native_viewer_id: int = 0
    native_viewer_key_file: str = ""
    screen_public_url: str = ""
    screen_frame_ttl: int = 10
    audio_awareness_enabled: bool = False
    audio_reactions_enabled: bool = True
    audio_url: str = "http://host.docker.internal:18769"
    audio_token: str = field(default="", repr=False)
    audio_timeout: float = 3.0
    stt_url: str = ""
    stt_api_key: str = field(default="", repr=False)
    stt_language: str = "auto"
    stt_context: str = "Bentebot, Bente."
    tts_danish_base_url: str = ""
    tts_base_url: str = ""
    tts_api_key: str = field(default="", repr=False)
    tts_model: str = ""
    tts_voice: str = ""
    tts_voices: str = ""
    tts_warm_enabled: bool = True
    tts_warm_interval: float = 120.0
    tts_warm_timeout: float = 60.0
    voice_profiles_enabled: bool = False
    voice_clone_policy: str = "server"
    voice_clone_speech_seconds: int = 45
    voice_clone_wait_seconds: int = 840
    allow_cloud_audio: bool = False
    interruption_words: int = 5
    voice_alone_timeout: float = 60.0
    turn_silence_ms: int = 800
    max_speakers: int = 6
    context_chars: int = 24000
    voice_context_chars: int = 8000
    retention_days: int = 90
    request_timeout: float = 60.0
    max_response_chars: int = 8000
    max_audio_bytes: int = 16_000_000
    max_tts_chars: int = 320
    health_port: int = 8080
    collect_text: bool = True
    music_enabled: bool = False
    music_downloads_enabled: bool = False
    music_library_path: str = "/music"
    music_metube_url: str = ""
    music_volume: float = 0.35

    @classmethod
    def from_env(cls) -> Settings:
        values = {}
        for name, info in cls.__dataclass_fields__.items():
            raw = os.getenv(f"BENTEBOT_{name.upper()}")
            if raw is None:
                continue
            default = info.default
            if isinstance(default, frozenset):
                values[name] = _ids(raw)
            elif isinstance(default, bool):
                values[name] = _bool(raw)
            elif isinstance(default, int):
                values[name] = int(raw)
            elif isinstance(default, float):
                values[name] = float(raw)
            else:
                values[name] = raw.strip()
        return cls(**values)

    def validate(self, *, login: bool = True) -> None:
        if not 0 <= self.music_volume <= 1:
            raise ValueError("music_volume must be between 0 and 1")
        if self.music_enabled and not self.music_library_path:
            raise ValueError("Music needs a library directory")
        if self.music_metube_url:
            parsed = urlsplit(self.music_metube_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("music_metube_url must be HTTP(S) without credentials or parameters")
        if self.music_enabled and self.music_downloads_enabled and not self.music_metube_url:
            raise ValueError("Music downloads require a MeTube URL")
        if (self.audio_awareness_enabled or self.tts_danish_base_url) and (
            not self.audio_url or len(self.audio_token) < 24
        ):
            raise ValueError("Audio awareness needs an API URL and a token of at least 24 characters")
        if not 0.1 <= self.audio_timeout <= 10:
            raise ValueError("audio_timeout must be between 0.1 and 10 seconds")
        if not 1 <= self.jupyter_timeout <= 120:
            raise ValueError("jupyter_timeout must be between 1 and 120 seconds")
        if self.jupyter_url:
            parsed = urlsplit(self.jupyter_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("jupyter_url must be HTTP(S) without credentials, query or fragment")
        if not 1 <= self.voice_alone_timeout <= 3600:
            raise ValueError("voice_alone_timeout must be between 1 and 3600 seconds")
        openai_native = {m.strip() for m in self.native_web_openai_models.split(",") if m.strip()}
        anthropic_native = {m.strip() for m in self.native_web_anthropic_models.split(",") if m.strip()}
        local_web = {m.strip() for m in self.web_tool_models.split(",") if m.strip()}
        if openai_native & anthropic_native or (openai_native | anthropic_native) & local_web:
            raise ValueError("Each web-enabled alias must select exactly one search backend")
        if self.vision_mode not in {"direct", "observer"}:
            raise ValueError("vision_mode must be direct or observer")
        if not 1 <= self.vision_timeout <= 60 or not 1 <= self.vision_requests_per_hour <= 1000:
            raise ValueError("Invalid vision timeout or hourly request limit")
        if not 1 <= self.web_tool_timeout <= 30 or not 1 <= self.web_max_tool_rounds <= 4:
            raise ValueError("Invalid web tool timeout or round limit")
        if self.web_tools_enabled and not (
            self.web_search_url and self.web_tool_models or openai_native or anthropic_native
        ):
            raise ValueError(
                "Web tools require a local search URL and model allowlist, or a native model allowlist"
            )
        if self.web_search_url:
            parsed = urlsplit(self.web_search_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("web_search_url must be an HTTP(S) URL without credentials or parameters")
        if not 30 <= self.tts_warm_interval <= 3600 or not 1 <= self.tts_warm_timeout <= 600:
            raise ValueError("Invalid TTS warm interval (30-3600 seconds) or timeout (1-600 seconds)")
        if not 30 <= self.voice_warm_interval <= 3600:
            raise ValueError("voice_warm_interval must be between 30 and 3600 seconds")
        if not 1 <= self.voice_warm_timeout <= 600:
            raise ValueError("voice_warm_timeout must be between 1 and 600 seconds")
        if self.chat_reasoning_effort not in {"", "none", "minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError("Invalid chat_reasoning_effort")
        if self.voice_clone_policy not in {"server", "individual"}:
            raise ValueError("voice_clone_policy must be server or individual")
        if not 30 <= self.voice_clone_speech_seconds <= 60 or not 60 <= self.voice_clone_wait_seconds <= 840:
            raise ValueError("Voice cloning needs 30-60 voiced seconds and a 60-840 second wait limit")
        if login and not self.discord_token:
            raise ValueError("Set BENTEBOT_DISCORD_TOKEN")
        if login and not self.guild_ids and not self.dm_user_ids:
            raise ValueError("Allow at least one guild or DM user explicitly")
        for name in (
            "interruption_words",
            "max_speakers",
            "context_chars",
            "voice_context_chars",
            "retention_days",
            "max_response_chars",
            "max_audio_bytes",
            "max_tts_chars",
            "chat_max_tokens",
            "screen_frame_ttl",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 200 <= self.turn_silence_ms <= 5000:
            raise ValueError("turn_silence_ms must be between 200 and 5000")
        if self.request_timeout <= 0 or not 0 <= self.health_port <= 65535:
            raise ValueError("Invalid request timeout or health port")
        if len(self.stt_context) > 1000 or "\x00" in self.stt_context:
            raise ValueError("stt_context must be at most 1000 characters without NUL")
        if self.generation_timeout <= 0:
            raise ValueError("generation_timeout must be positive")
        if self.voice_first_content_timeout <= 0:
            raise ValueError("voice_first_content_timeout must be positive")
        if self.screen_enabled and not self.health_port:
            raise ValueError("Screen sharing requires a health port")
        if self.native_viewer_enabled and (
            not self.screen_enabled or self.native_viewer_id <= 0 or not self.native_viewer_key_file
        ):
            raise ValueError(
                "Native viewing requires screen_enabled, native_viewer_id and native_viewer_key_file"
            )
        if self.screen_public_url:
            parsed = urlsplit(self.screen_public_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("screen_public_url must be an HTTP(S) URL without credentials or parameters")
        for name in ("chat_base_url", "tts_base_url", "tts_danish_base_url", "audio_url", "stt_url"):
            value = getattr(self, name)
            if not value:
                continue
            parsed = urlsplit(value)
            schemes = {"ws", "wss"} if name == "stt_url" else {"http", "https"}
            if parsed.scheme not in schemes or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError(f"Invalid {name}; use a URL without embedded credentials")
            if parsed.query or parsed.fragment:
                raise ValueError(f"{name} must not contain query parameters or fragments")
            # Cloud audio is an explicit opt-in; there are no paid automatic fallbacks.
            if name != "chat_base_url" and not self.allow_cloud_audio:
                import ipaddress

                host = parsed.hostname
                try:
                    local = ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
                except ValueError:
                    local = (
                        host == "localhost"
                        or "." not in host
                        or host.endswith((".local", ".lan", ".internal"))
                    )
                if not local:
                    raise ValueError(
                        "Cloud audio requires BENTEBOT_ALLOW_CLOUD_AUDIO=true; verify it is free"
                    )

    @property
    def voice_configured(self) -> bool:
        return bool(self.stt_url and self.tts_base_url and self.tts_model and self.tts_voice)
