"""A narrow guard against generic appended service/readiness questions."""

import re

_HELP = (
    r"(?:is there |was there )?anything else(?: "
    r"(?:i can (?:help(?: you)?(?: out)? with|do for you)|"
    r"you (?:need(?: help with)?|want(?: help with)?|would like(?: help with| to (?:know|discuss|talk about|ask)))|"
    r"you['’]d like(?: help with| to (?:know|discuss|talk about|ask))))?"
    r"(?: (?:today|at all|right now))?"
)
_READY = (
    r"(?:are you |would you be )?ready (?:to |for )?(?:test(?:ing)?|try(?:ing)?)"
    r"(?: (?:it|this|that|things|the bot|screen sharing))?(?: out)?(?: now)?"
)
_DANISH = (
    r"(?:er der )?(?:noget andet|andet)(?: jeg kan hjælpe(?: dig)? med| du har brug for hjælp til|"
    r" du (?:vil|gerne vil) (?:vide|spørge om))?|"
    r"(?:er du )?klar til at (?:teste|prøve)(?: det(?: af)?)?(?: nu)?"
)
_TAIL = re.compile(rf"(?:^|(?<=[.!?,;:\u2014]))\s*(?:{_HELP}|{_READY}|{_DANISH})\s*[?!.]*\s*$", re.IGNORECASE)


def without_stock_tail(text: str) -> str:
    while match := _TAIL.search(text):
        text = text[: match.start()].rstrip(" ,;:\u2014")
    return text


class VoiceFollowupFilter:
    def __init__(self):
        self.has_content = False

    def __call__(self, sentence: str) -> str:
        cleaned = without_stock_tail(sentence)
        # A standalone question may be the actual point of the reply. Only
        # discard an entire sentence after substantive speech has been queued.
        if not cleaned and not self.has_content:
            cleaned = sentence
        if cleaned:
            self.has_content = True
        return cleaned
