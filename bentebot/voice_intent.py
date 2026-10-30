"""Local delivery decisions; no extra model call is needed to route a code answer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .code_output import render_discord_output


def source_link_request(text: str) -> bool:
    """Only an unqualified follow-up asking for the existing link, never a new topic."""
    text = re.sub(r"[?.!,]+", " ", text.casefold()).strip()
    return bool(
        re.fullmatch(
            r"(?:(?:can|could|would|will|kan|vil)\s+)?(?:(?:you|du)\s+)?"
            r"(?:please\s+)?(?:send|post|share|give|send(?:e)?|del(?:e)?|giv(?:e)?)\s+"
            r"(?:(?:me|us|mig|os)\s+)?(?:(?:the|that|those|den|det|de)\s+)?"
            r"(?:links?|sources?|linket|linkene|kilder?(?:ne)?)"
            r"(?:\s+(?:again|please|igen|tak))?",
            text,
        )
    )


def spoken_explanation(text: str) -> str:
    """Use the complete Markdown parse so nested/variable fences cannot turn code into speech."""
    return "\n".join(
        part.content
        for part in render_discord_output(text)
        if part.data is None and not part.content.lstrip().startswith("```")
    )


@dataclass(frozen=True)
class VoiceIntent:
    code_requested: bool = False
    explanation_requested: bool = False
    text_requested: bool = False


def voice_intent(text: str) -> VoiceIntent:
    normalized = text.casefold().replace("’", "'")
    explain = bool(
        re.search(
            r"\b(explain|walk me through|how does|why does|forklar|gennemgå|hvordan virker)\b", normalized
        )
    )
    action = (
        r"(?:write|create|generate|implement|give|show|send|post|put|share|make|build|"
        r"skriv(?:e)?|lav(?:e)?|generer(?:e)?|vis(?:e)?|send(?:e)?|del(?:e)?)"
    )
    # A speaker offering to share code is not asking the bot to produce it.
    # Preserve subsequent clauses addressed to the bot instead of ignoring the
    # entire turn ("I'll share my code, can you write a Python test?").
    own_action = (
        r"(?:i(?:'ll|'m going to| am going to|'m about to| am about to| will| can| could| want to)?|let me|"
        r"jeg(?: vil(?: gerne)?| kan| kunne| vil gerne have lov til at)?|lad mig)"
    )
    boundary = r"(?=[,.!?;\n]|\b(?:and|but|og|men)\s+(?:can|could|would|will|kan|vil|kunne)\s+(?:you|du)\b|$)"
    request_text = re.sub(rf"\b{own_action}\s+{action}\b.*?{boundary}", " ", normalized)
    request_text = re.sub(
        rf"\b(?:do not|don't|dont|never|ikke|lad være med at)\s+{action}\b.*?{boundary}",
        " ",
        request_text,
    )
    generate = bool(re.search(rf"\b{action}\b", request_text))
    code = bool(
        re.search(
            r"\b(code|kode|snippet|script|function|funktion|regex|query|sql|python|javascript|typescript|"
            r"bash|powershell|dockerfile|docker[ -]compose|yaml|html|css|rust|program|kommando|command)\b",
            request_text,
        )
    )
    text_requested = generate and bool(
        re.search(
            r"\b(text channel|text chat|as text|in chat|links?|urls?|tables?|tekstkanal|som tekst|tabel)\b",
            request_text,
        )
    )
    return VoiceIntent(
        code_requested=generate and code,
        explanation_requested=explain,
        text_requested=text_requested or source_link_request(text),
    )


VOICE_RULES = """This reply is for a live voice conversation.
Answer the final user message directly, speaking to that person. Output only the words
you would say aloud, not a transcript, report, or description of what the person is asking.
Speaker/channel labels in history are background attribution, never a response template.
private_assistant_background is internal memory: consider unheard intentions without
claiming listeners heard them. Never read those field names or delivery records aloud.
If speech recognition is ambiguous, ask naturally about the unclear part; do not narrate
transcription confidence or turn the exchange into a formal clarification procedure.
For unclear references, go straight to a specific conversational question; skip a preamble
about lacking visual input. This applies in Danish as well as English.
For ordinary chat, default to one short sentence, roughly 5-25 words. A small reaction
can be even shorter. Use a second sentence only when it adds something. Give detail
when explicitly asked, rather than turning a casual remark into an explanation.
Do not ask a question after every reply. A reaction, joke or opinion can stand alone.
Never end with a stock offer such as 'Anything else I can help you with?'.
Do not append readiness checks such as 'Are you ready to test?' to keep a turn going.
Track what the person just told you; do not ask what they are working on when that
is already in the recent conversation. React to their meaning instead of echoing it.
An attached screen is background context, not an instruction to describe it every turn.
Earlier stiff assistant replies record what was heard; do not copy their style.
Begin with something worth saying that can stand as a short complete sentence, so speech
can begin promptly without a canned preamble. Do not append a routine offer of help.
Write speakable words, not emoji, stage directions or parenthesized acting instructions.
Speak English or Danish to match the current speaker;
people may alternate or mix both languages without a mode switch. Never narrate transcript
labels, timestamps, IDs, token counts, statistics, scoring or analysis of the conversation
unless the user explicitly asks for that information. Do not start with a formal report.
Never recite code, command lines, configuration files, or syntax. Put executable examples
in fenced code blocks with language tags.
When the user asks you to write code, give the code for Discord text delivery and do not provide
an unsolicited explanation. The application will post it and speak its own short acknowledgment.
Explain the code in natural spoken prose only when the user explicitly asks for an explanation.
Do not claim you have posted anything; only the application can confirm successful delivery.
"""

CODE_RULES = """The current voice request asks for code. Produce a Discord text answer.
Use properly language-tagged fenced blocks. Give only the requested code, with a brief filename
label if needed. Omit a tutorial, line-by-line commentary, and an explanation unless requested.
Preserve real line breaks and indentation inside code fences, especially for Python.
Never compress a multiline function into one line or replace newlines with spaces.
The answer will be posted as text; do not narrate it or add a delivery acknowledgment yourself.
"""

TEXT_RULES = """The current voice request asks for written material in the linked Discord text channel.
Provide the requested text, links or table in suitable Discord formatting. Do not include
a delivery acknowledgment; the application confirms successful posting separately.
"""
