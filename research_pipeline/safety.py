"""Retrieved text is untrusted input.

Papers and preprints have been found carrying hidden instructions aimed at language models. The
pipeline's structural defences are that it holds no write tools, wraps passages as data, and decodes
every model output under a schema. This scan is the cheap extra layer: a passage that addresses the
reader-model is dropped from the evidence pool and recorded in the run log.

It is an English-language pattern list, so it catches the known, careless cases and not a
determined adversary. The structural defences are the ones that matter.
"""

from __future__ import annotations

import re

INJECTION = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above|preceding) (?:instructions|prompts?|text)"
    r"|disregard (?:all |any |the )?(?:previous|prior|above) "
    r"|(?:as|you are) an? (?:ai|llm|large language model|language model)\b"
    r"|\b(?:system|developer) prompt\b"
    r"|do not (?:highlight|mention|report) (?:any )?(?:negatives|weaknesses|limitations)"
    r"|give (?:this paper |it )?a positive review"
    r"|recommend accept(?:ance|ing)? (?:of )?this (?:paper|manuscript)"
    r"|<\|im_start\|>|<\|im_end\|>|\[/?INST\]",
    re.IGNORECASE,
)
# Zero-width, bidirectional-control and invisible formatting characters, used to hide text from
# human readers. Written as escapes so the pattern itself is visible in review.
HIDDEN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
# PDF extraction produces the odd stray zero-width space; a handful is noise, not concealment.
HIDDEN_TOLERANCE = 3


def scan(text: str) -> list[str]:
    """Flags for one passage. An empty list means nothing suspicious was found."""
    flags = []
    if INJECTION.search(text):
        flags.append("prompt_injection_pattern")
    if scan_hidden(text):
        flags.append("hidden_characters")
    return flags


def scan_hidden(text: str) -> str | None:
    """Describe hidden characters in raw text, or None. Call this before `clean`."""
    count = len(HIDDEN.findall(text))
    return f"{count} hidden characters" if count > HIDDEN_TOLERANCE else None


def clean(text: str) -> str:
    return HIDDEN.sub("", text)
