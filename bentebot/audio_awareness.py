"""Nonblocking, bounded per-caller sound/affect context. No recording persistence."""

from __future__ import annotations

import asyncio
import audioop
import json
import time
from collections import Counter

import httpx

SOUNDS = {"laugh", "cough", "sneeze", "gasp", "sigh", "scream", "cry"}
TONES = {"angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"}
TONE_STATUS = {
    "low_volume",
    "not_requested",
    "insufficient_audio",
    "no_speech",
    "low_confidence",
    "low_margin",
    "candidate",
}


class AudioAwareness:
    def __init__(self, settings, changed, *, client=None, clock=time.monotonic):
        self.settings, self.changed, self.clock = settings, changed, clock
        self.client = client or httpx.AsyncClient(timeout=settings.audio_timeout, follow_redirects=False)
        self.own_client = client is None
        self.buffers = {}
        self.voiced = Counter()
        self.revisions = {}
        self.serial = 0
        self.tone_at = {}
        self.cues = {}
        self.diagnostics = {}
        self.queue = asyncio.Queue(maxsize=2)
        self.runner = None
        self.closed = False
        self.reaction = None
        self.reacted_at = float("-inf")
        self.stats = Counter()
        self.device = "unknown"

    def observe(self, user_id, frame, voiced):
        if self.closed:
            return
        if user_id not in self.buffers and len(self.buffers) >= self.settings.max_speakers:
            self.stats["dropped"] += 1
            return
        if user_id not in self.revisions:
            self.serial += 1
            self.revisions[user_id] = self.serial
        buffer = self.buffers.setdefault(user_id, bytearray())
        buffer.extend(frame)
        self.voiced[user_id] += int(voiced)
        if len(buffer) < 96000:
            return
        pcm = bytes(buffer[:96000])
        del buffer[:96000]
        speech_frames = self.voiced.pop(user_id, 0)
        if audioop.rms(pcm, 2) < 90:
            return
        now = self.clock()
        tone = speech_frames >= 20 and now - self.tone_at.get(user_id, float("-inf")) >= 6
        if tone:
            self.tone_at[user_id] = now
        if self.queue.full():
            self.queue.get_nowait()
            self.stats["dropped"] += 1
        self.queue.put_nowait((user_id, self.revisions[user_id], pcm, tone, now))
        if self.runner is None or self.runner.done():
            self.runner = asyncio.create_task(self._run(), name="audio-awareness")

    async def _run(self):
        while not self.closed:
            user_id, revision, pcm, tone, observed = await self.queue.get()
            if self.clock() - observed > 5 or revision != self.revisions.get(user_id):
                continue
            try:
                self.stats["requests"] += 1
                if tone:
                    self.stats["tone_requests"] += 1
                response = await self.client.post(
                    self.settings.audio_url.rstrip("/") + "/v1/analyze",
                    content=pcm,
                    headers={
                        "Authorization": "Bearer " + self.settings.audio_token,
                        "Content-Type": "application/octet-stream",
                        "X-Analyze-Tone": "1" if tone else "0",
                    },
                )
                response.raise_for_status()
                if len(response.content) > 16000:
                    raise ValueError("Audio result too large")
                result = response.json()
                if self.closed or revision != self.revisions.get(user_id) or self.clock() - observed > 8:
                    continue
                self.device = result.get("device") if result.get("device") in {"cpu", "cuda"} else "unknown"
                events = [
                    {"label": e["label"], "score": float(e["score"])}
                    for e in result.get("events", [])
                    if isinstance(e, dict)
                    and e.get("label") in SOUNDS
                    and type(e.get("score")) in {int, float}
                    and 0.75 <= e["score"] <= 1
                ]
                affect = result.get("tone", "unknown")
                score = result.get("tone_score", 0)
                if affect not in TONES or type(score) not in {int, float} or not 0.7 <= score <= 1:
                    affect, score = "unknown", 0
                previous = self.cues.get(user_id, {})
                prior_tone_at = previous.get("tone_at", float("-inf"))
                # One high score is not a stable mood. Require agreement across two
                # analyzed windows; unknown does not become an asserted emotion.
                stable = (
                    affect
                    if affect == previous.get("candidate") and observed - prior_tone_at < 12
                    else "unknown"
                )
                cue = {
                    "events": events,
                    "tone": stable,
                    "candidate": affect,
                    "tone_score": score,
                    "tone_at": observed if tone else prior_tone_at,
                    "at": observed,
                    "speech_score": result.get("speech_score", 1),
                }
                if not tone and previous and observed - prior_tone_at < 12:
                    cue.update(
                        tone=previous["tone"],
                        candidate=previous["candidate"],
                        tone_score=previous["tone_score"],
                    )
                self.cues[user_id] = cue
                diagnostic = self.diagnostics.setdefault(user_id, {})
                top = result.get("top_sound")
                if (
                    isinstance(top, dict)
                    and top.get("label") in SOUNDS
                    and type(top.get("score")) in {int, float}
                    and 0 <= top["score"] <= 1
                ):
                    diagnostic["sound"] = {"at": observed, "label": top["label"], "score": top["score"]}
                if events:
                    self.stats["sound_detections"] += 1
                    diagnostic["detected"] = dict(max(events, key=lambda e: e["score"]), at=observed)
                if tone:
                    self.stats["tone_analyses"] += 1
                    gate = result.get("tone_status")
                    candidate = result.get("tone_candidate")
                    diagnostic["tone"] = {
                        "at": observed,
                        "status": gate if gate in TONE_STATUS else "unreported",
                        "candidate": candidate if candidate in TONES else "unknown",
                        "score": self._score(result.get("tone_score")),
                        "margin": self._score(result.get("tone_margin")),
                        "speech_score": self._score(result.get("speech_score")),
                        "stable": stable,
                    }
                    if stable not in {"unknown", "other"}:
                        self.stats["tone_confirmations"] += 1
                self.stats["analyses"] += 1
                self.stats["last_ms"] = result.get("elapsed_ms", 0)
                if events and self.settings.audio_reactions_enabled:
                    event = max(events, key=lambda e: e["score"])
                    if event["score"] >= 0.85:
                        self.reaction = (user_id, event["label"], observed, revision)
                self.changed.set()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Audio awareness is optional; never masquerade as an STT failure.
                self.stats["unavailable"] += 1

    def context(self, audience, names):
        cues = []
        for user_id, cue in list(self.cues.items()):
            if self.clock() - cue["at"] > 15:
                self.cues.pop(user_id, None)
            elif user_id in audience.user_ids and (
                cue["events"] or cue["tone"] not in {"neutral", "unknown", "other"}
            ):
                cues.append(
                    {
                        "speaker": names.get(user_id, "a caller"),
                        "sounds": cue["events"],
                        "possible_tone": cue["tone"],
                        "age_seconds": round(self.clock() - cue["at"], 1),
                    }
                )
        if not cues:
            return []
        return [
            {
                "role": "system",
                "content": "Uncertain audio cues, not transcripts or facts about anyone's mood. Combine with actual dialogue; "
                "adjust length/timing/humour subtly if useful. Do not announce labels, diagnose feelings, praise "
                "or validate automatically, or treat a sound as a spoken question. Data: " + json.dumps(cues),
            }
        ]

    def take_reaction(self):
        reaction, self.reaction = self.reaction, None
        if reaction is None:
            return None
        user_id, label, observed, revision = reaction
        if (
            self.clock() - observed > 8
            or revision != self.revisions.get(user_id)
            or self.clock() - self.reacted_at < 45
        ):
            return None
        self.reacted_at = self.clock()
        return user_id, label, observed

    def drop(self, user_id):
        self.revisions.pop(user_id, None)
        self.buffers.pop(user_id, None)
        self.voiced.pop(user_id, None)
        self.tone_at.pop(user_id, None)
        self.cues.pop(user_id, None)
        self.diagnostics.pop(user_id, None)

    def suspend(self):
        for user_id in list(self.buffers):
            self.drop(user_id)
        self.reaction = None
        while not self.queue.empty():
            self.queue.get_nowait()

    def status(self):
        return {"audio_" + key: value for key, value in self.stats.items()} | {
            "audio_queued": self.queue.qsize(),
            "audio_yawn": "unsupported",
            "audio_device": self.device,
        }

    @staticmethod
    def _score(value):
        return round(value, 4) if type(value) in {int, float} and 0 <= value <= 1 else 0

    def speaker_status(self, user_id):
        now = self.clock()
        state = {}
        if cue := self.cues.get(user_id):
            if now - cue["at"] <= 15:
                state.update(
                    your_possible_tone=cue["tone"],
                    your_sound_events=",".join(e["label"] for e in cue["events"]) or "none",
                    your_audio_cue_age_seconds=round(now - cue["at"], 1),
                )
        # Diagnostics last up to one minute in call memory only. They never enter
        # the conversation prompt or expose another caller's tentative labels.
        diagnostic = self.diagnostics.get(user_id, {})
        if tone := diagnostic.get("tone"):
            if now - tone["at"] <= 60:
                state.update(
                    your_tone_gate=tone["status"],
                    your_tone_candidate=tone["candidate"],
                    your_tone_confidence=tone["score"],
                    your_tone_margin=tone["margin"],
                    your_tone_speech_score=tone["speech_score"],
                    your_tone_analysis_age_seconds=round(now - tone["at"], 1),
                    your_tone_waiting_for_agreement=tone["status"] == "candidate"
                    and tone["stable"] == "unknown",
                )
        for key, prefix in (("sound", "your_strongest_sound"), ("detected", "your_last_confirmed_sound")):
            if event := diagnostic.get(key):
                if now - event["at"] <= 60:
                    state[prefix] = event["label"]
                    state[prefix + "_score"] = event["score"]
                    state[prefix + "_age_seconds"] = round(now - event["at"], 1)
        return state

    async def close(self):
        self.closed = True
        self.suspend()
        if self.runner:
            self.runner.cancel()
            await asyncio.gather(self.runner, return_exceptions=True)
        if self.own_client:
            await self.client.aclose()
