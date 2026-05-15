"""Routing protocol — model↔runner contract for in-text routing decisions.

The model emits a self-closing XML tag at the end of its response when it
wants to fire a routing decision:

    Got it, I'll send that to you now. <route exit="xp_send_confirmation" />

    Sorry — I can't help with that. <route interrupt="int_escalate_human" />

    Speak naturally. <route exit="xp_done" var1="value" var2="value" />

Variants:
- `<route exit="..." [captures...] />` — take an exit.
- `<route interrupt="..." />` — trigger an interrupt.
- No tag at all — stay in the current flow.

Format guarantees enforced at parse time:
- Self-closing tag (`<route ... />`). No closing `</route>`.
- Single tag per response. If multiple, first wins (logged).
- `exit` and `interrupt` are mutually exclusive. If both present, ignore.
- Unknown attributes on `exit` form become string captures (passed as
  llm-method assigns to the dispatcher).
- Malformed tags are silently ignored — treated as "stay".

Reserved sentinels stripped from the user-facing output (in addition to
`<route ... />`):
- `<think>...</think>` — reasoning scaffolding the LLM is encouraged to
  emit but the user shouldn't hear. Mirrors a common Claude / Gemini
  prompting convention. Contents are discarded; no dispatch side-effect.

Any OTHER tag-shaped text (`<strong>`, `<VERIFICATION>`, etc.) is left
intact and flows through to TTS / the text caller. We do not auto-strip
arbitrary tags — that would create surprising losses for any spec that
uses tag-shaped patterns for non-reasoning purposes. If a future case
arises where stripping is needed, add it as another named reserved
sentinel here, not as a blanket rule.

The streaming voice path uses `RouteTagFrameProcessor` (in processor.py)
to remove these sentinels from the TTS stream as they're emitted. Text
mode parses the full response string after generation. Both call into
this module's helpers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# A self-closing route tag. Captures the attribute body for downstream parse.
# - Allows whitespace inside the tag (around attributes and before />).
# - Greedy match would over-consume on multi-tag input; we use non-greedy
#   and a `findall` so the caller can decide first-wins behavior.
_TAG_RE = re.compile(
    r"<\s*route\b([^>]*?)/\s*>",
    re.DOTALL,
)

# Reasoning scaffolding the user shouldn't hear. Discarded entirely.
# Block form `<think>...</think>` only — the closing tag is required so we
# never accidentally swallow a stray `<think` on its own.
_THINK_RE = re.compile(r"<\s*think\b[^>]*>.*?<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE)

# attr="value" with double-quoted values. Single-quoted and unquoted values
# are not supported — the prompt protocol always uses double quotes, and
# being strict here makes malformed tags fail closed (treated as stay)
# rather than producing surprising routes.
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


@dataclass(frozen=True)
class RouteTag:
    """Parsed contents of a `<route ... />` tag.

    Exactly one of `exit` or `interrupt` is set on a well-formed tag. Both
    None means the tag was malformed or empty — caller should treat as
    "stay" (no routing).
    """

    exit: str | None = None
    interrupt: str | None = None
    captures: dict[str, str] | None = None  # extra attributes on exit tags

    @property
    def is_valid(self) -> bool:
        # Exactly one of exit/interrupt must be set.
        return bool(self.exit) != bool(self.interrupt)


def strip_think_blocks(text: str) -> str:
    """Remove any `<think>...</think>` blocks (reserved reasoning sentinel).
    See module docstring for the convention. Idempotent."""
    return _THINK_RE.sub("", text)


def find_tag(text: str) -> tuple[str, RouteTag | None]:
    """Locate the first route tag in `text`. Return (text_without_tag, parsed).

    Also strips any `<think>...</think>` blocks (reserved reasoning sentinel)
    so the returned text is what the user should see / hear.

    First-wins: additional route tags after the first are left in the text.
    They will be ignored downstream and logged at parse time.
    """
    text = strip_think_blocks(text)
    match = _TAG_RE.search(text)
    if match is None:
        return text, None

    parsed = _parse_attrs(match.group(1))
    # Strip the tag and any trailing whitespace immediately preceding it on
    # the same line. Callers want a clean reply, not "..., I'll send that\n  ".
    before = text[: match.start()].rstrip(" \t")
    after = text[match.end() :]
    # If the tag was on its own line, also drop the now-orphan newline.
    if before.endswith("\n"):
        before = before.rstrip("\n").rstrip(" \t")
    cleaned = before + after
    return cleaned, parsed


def parse_tag_body(body: str) -> RouteTag | None:
    """Parse a tag attribute string like `exit="xp_X" date="2026-05-20"`.

    Returns None if the result wouldn't be a valid tag (neither exit nor
    interrupt, or both). Used by the streaming stripper after it has
    accumulated the tag bytes.
    """
    parsed = _parse_attrs(body)
    return parsed if parsed.is_valid else None


def _parse_attrs(body: str) -> RouteTag:
    attrs = {k: v for k, v in _ATTR_RE.findall(body)}
    exit_id = attrs.pop("exit", None)
    interrupt_id = attrs.pop("interrupt", None)
    return RouteTag(
        exit=exit_id,
        interrupt=interrupt_id,
        captures=attrs if attrs else None,
    )


def to_llm_results(tag: RouteTag) -> dict[str, dict]:
    """Adapt a parsed RouteTag into the `llm_results` dict shape that
    `routing.resolve` consumes. Mirrors the legacy tool-call payload.
    """
    if tag.exit:
        payload: dict = {"exit_path_id": tag.exit}
        if tag.captures:
            payload.update(tag.captures)
        return {"take_exit_path": payload}
    if tag.interrupt:
        return {"trigger_interrupt": {"interrupt_flow_id": tag.interrupt}}
    return {}
