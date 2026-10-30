from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime

from .domain import Audience, Event, Source
from .permissions import AccessPolicy
from .persona import DEFAULT_PERSONALITY as PERSONALITY  # noqa: F401
from .persona import PersonaStore
from .storage import Store
from .voice_style import without_stock_tail


class Memory:
    def __init__(
        self,
        store: Store,
        policy: AccessPolicy,
        max_chars: int,
        source_label: Callable[[Source], str] | None = None,
        *,
        voice_max_chars: int | None = None,
    ):
        self.store, self.policy, self.max_chars = store, policy, max_chars
        self.personas = PersonaStore(store)
        self.voice_max_chars = voice_max_chars
        self.source_label = source_label or (lambda source: str(source.channel_id))

    def context_budget(self, *, voice: bool = False) -> int:
        """Bound active voice history separately; durable history stays intact."""
        return min(self.max_chars, self.voice_max_chars) if voice and self.voice_max_chars else self.max_chars

    async def search(self, audience: Audience, query: str, limit: int = 10) -> list[Event]:
        """Search compressed history without exposing hits, counts or snippets from denied sources."""
        if limit <= 0:
            return []
        found = []
        permitted: dict[Source, bool] = {}
        for event in self.store.search(query, audience.guild_id, audience.user_ids, limit=200):
            origins = tuple(dict.fromkeys((event.source, *event.provenance)))
            for origin in origins:
                if origin not in permitted:
                    permitted[origin] = await self.policy.allows(origin, audience)
            if all(permitted[origin] for origin in origins):
                found.append(event)
                if len(found) >= min(limit, 20):
                    break
        return found

    async def paused_reply(self, event_id: str, audience: Audience) -> tuple[dict | None, tuple[Source, ...]]:
        """Recover a paused turn through current permissions, even after long side talk."""
        event = self.store.event(event_id)
        if event is None or event.kind != "assistant" or not event.interrupted:
            return None, ()
        origins = tuple(dict.fromkeys((event.source, *event.provenance)))
        if not await self.policy.allows_all(origins, audience):
            return None, ()
        budget = max(128, min(8000, self.max_chars // 2))
        heard = event.spoken[-min(2000, budget // 3) :]
        # A small explicit overlap helps the model resume a thought rather than
        # interpreting the estimated heard text as a hard word-level cursor.
        last_sentence = re.split(r"(?<=[.!?。！？])\s+", heard.strip())[-1] if heard.strip() else ""
        overlap = " ".join(last_sentence.split()[-40:])[-min(320, budget // 4) :]
        draft = event.generated[-max(1, budget - len(heard) - len(overlap) - 250) :]
        return {
            "role": "user",
            "content": json.dumps(
                {
                    "paused_reply_background": {
                        "heard_so_far_estimated": heard,
                        "resume_overlap_excerpt": overlap,
                        "previous_generated_draft_not_all_heard": draft,
                    }
                },
                ensure_ascii=False,
            ),
        }, origins

    async def context(
        self,
        audience: Audience,
        query: str = "",
        *,
        voice: bool = False,
        persona_channel_id: int | None = None,
    ) -> tuple[list[dict], tuple[Source, ...]]:
        selected: list[tuple[dict, tuple[Source, ...]]] = []
        retrieved = await self.search(audience, query, limit=10) if query.strip() else []
        max_chars = self.context_budget(voice=voice)
        budget = max_chars
        permitted: dict[Source, bool] = {}
        selected_ids: set[str] = set()
        events = self.store.recent(audience.guild_id, audience.user_ids)
        for event in reversed(events):
            origins = tuple(dict.fromkeys((event.source, *event.provenance)))
            for origin in origins:
                if origin not in permitted:
                    permitted[origin] = await self.policy.allows(origin, audience)
            if not all(permitted[origin] for origin in origins):
                continue
            text, role = (
                self.render_voice(event)
                if voice
                else (self.render(event), "assistant" if event.kind == "assistant" else "user")
            )
            if not text:
                continue
            # Keep the newest event even if it is large; never send unbounded history.
            if len(text) > budget and selected:
                break
            text = text[-budget:]
            selected.append(({"role": role, "content": text}, origins))
            selected_ids.add(event.id)
            budget -= len(text)
            if budget <= 0:
                break
        selected.reverse()
        retrieved = [event for event in retrieved if event.id not in selected_ids]
        if retrieved:
            # Only reserve retrieval space when useful hits are actually absent from the
            # ordinary recent context. A query matching the current turn does not reduce it.
            recent_budget = max_chars - min(max_chars // 3, 8000)
            used = sum(len(message["content"]) for message, _ in selected)
            while used > recent_budget and len(selected) > 1:
                message, _ = selected.pop(0)
                used -= len(message["content"])
            if used > recent_budget and selected:
                selected[0][0]["content"] = selected[0][0]["content"][-recent_budget:]
        older = []
        remaining = max_chars - sum(len(message["content"]) for message, _ in selected)
        # Rank selects useful hits; chronology within that background block preserves their dates.
        for event in retrieved:
            if event.id in selected_ids or remaining <= 0:
                continue
            origins = tuple(dict.fromkeys((event.source, *event.provenance)))
            # Recheck after the search await: a permission change must not turn a previous hit
            # into prompt context merely because it was allowed earlier in this method.
            if not await self.policy.allows_all(origins, audience):
                continue
            rendered, role = (
                self.render_voice(event)
                if voice
                else (self.render(event), "assistant" if event.kind == "assistant" else "user")
            )
            if not rendered:
                continue
            text = "[Retrieved earlier context; not a new message]\n" + rendered
            if len(text) > remaining:
                if older:
                    continue
                text = text[:remaining]
            older.append((event.created_at, {"role": role, "content": text}, origins))
            selected_ids.add(event.id)
            remaining -= len(text)
        older.sort(key=lambda item: item[0])
        selected = [(message, origins) for _, message, origins in older] + selected
        sources = tuple(dict.fromkeys(source for _, origins in selected for source in origins))
        return [
            {
                "role": "system",
                "content": self.personas.prompt(
                    audience.guild_id, persona_channel_id or audience.destination_id
                ),
            },
            *(message for message, _ in selected),
        ], sources

    def render_voice(self, event: Event) -> tuple[str, str]:
        """Conversational examples contain actual words, never delivery bookkeeping.

        Permission/provenance checks still use the original IDs outside the prompt.
        Unheard intentions remain background data, never an assistant turn presented
        as delivered speech. Original database records remain untouched.
        """
        if event.kind != "assistant":
            label = self.source_label(event.source)
            data = {"speaker": event.author_name, "message": event.content}
            if label and not re.search(r"\d{15,22}", label):
                data["channel"] = label
            return json.dumps(data, ensure_ascii=False), "user"
        spoken = (event.spoken or event.content) if event.delivery == "voice" else event.content
        # Do not teach the model to imitate old accidental bookkeeping readouts.
        if re.search(
            r"Spoken aloud \(estimated\):|\[Bentebot reply in |\b(?:channel_id|started_at|ended_at)[\"']?\s*[:=]|"
            r"(?:sent|text)\s+in\s+(?:(?:the|discord)\s+)*channel\s+\d{15,22}",
            spoken,
            re.IGNORECASE,
        ):
            return "", "assistant"
        shared = event.written if event.written != spoken else ""
        generated = " ".join(event.generated.split())
        heard = " ".join(spoken.split())
        draft = ""
        if event.delivery == "none":
            spoken, shared, draft = "", "", event.generated
        elif generated and event.written != event.generated:
            if generated.startswith(heard):
                draft = generated[len(heard) :].strip()
            else:
                draft = event.generated
        draft = without_stock_tail(draft).strip()
        if draft or event.interrupted:
            return json.dumps(
                {
                    "private_assistant_background": {
                        "heard": spoken,
                        "shared_text": shared,
                        "unheard_intention": draft,
                        "interrupted": event.interrupted,
                    }
                },
                ensure_ascii=False,
            ), "user"
        return "\n\n".join(part for part in (spoken, shared) if part), "assistant"

    def render(self, event: Event) -> str:
        if event.kind == "assistant":
            origin = f"[Bentebot reply in {self.source_label(event.source)}]"
            if event.delivery == "voice":
                parts = [origin]
                if event.spoken:
                    parts.append(f"Spoken aloud (estimated): {event.spoken}")
                if event.written:
                    parts.append(f"Sent as text in channel {event.written_channel_id}: {event.written}")
                matched = event.generated.startswith(event.spoken)
                draft = event.generated[len(event.spoken) :] if matched else event.generated
                if event.written == event.generated:
                    draft = ""
                if draft.strip():
                    label = (
                        "Unspoken draft, never delivered to listeners"
                        if matched
                        else "Original draft; compare with spoken text above; the remainder was never spoken"
                    )
                    parts.append(f"{label}: {draft}")
                if event.interrupted:
                    parts.append("This response was interrupted.")
                return "\n".join(parts)
            if event.delivery == "none":
                return (
                    f"{origin}\nUnsent draft, never delivered: {event.generated}" if event.generated else ""
                )
            text = f"{origin}\n{event.content}"
            if event.interrupted:
                text += f"\nOriginal draft; only the text above was delivered: {event.generated}"
            return text
        return json.dumps(
            {
                "speaker": event.author_name,
                "user_id": str(event.author_id),
                "source": event.kind,
                "channel": self.source_label(event.source),
                "channel_id": str(event.source.channel_id),
                "started_at": datetime.fromtimestamp(event.created_at, UTC).isoformat(),
                "ended_at": datetime.fromtimestamp(event.ended_at, UTC).isoformat()
                if event.ended_at
                else None,
                "message": event.content,
            },
            ensure_ascii=False,
        )
