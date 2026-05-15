"""Parser tests for the in-text routing protocol.

Pure parser, no dispatcher / no LLM. These tests cover the wire-format
contract: what tag shapes are accepted, what's rejected, and how the tag
is stripped from the conversational text.
"""

from __future__ import annotations

from uxflows_runner.dispatcher.routing_protocol import (
    RouteTag,
    find_tag,
    parse_tag_body,
    strip_think_blocks,
    to_llm_results,
)


# --------------------------------------------------------------------------
# find_tag — full-string extraction
# --------------------------------------------------------------------------


def test_no_tag_returns_text_unchanged():
    text = "Just a conversational reply, no routing."
    cleaned, tag = find_tag(text)
    assert tag is None
    assert cleaned == text


def test_exit_tag_at_end_strips_cleanly():
    text = 'Got it, I will send that now. <route exit="xp_send" />'
    cleaned, tag = find_tag(text)
    assert tag == RouteTag(exit="xp_send", interrupt=None, captures=None)
    assert cleaned == "Got it, I will send that now."


def test_interrupt_tag_at_end_strips_cleanly():
    text = 'I cannot help with that. <route interrupt="int_escalate" />'
    cleaned, tag = find_tag(text)
    assert tag == RouteTag(exit=None, interrupt="int_escalate", captures=None)
    assert cleaned == "I cannot help with that."


def test_tag_on_its_own_line():
    text = "Have a great day.\n<route exit=\"xp_end\" />"
    cleaned, tag = find_tag(text)
    assert tag.exit == "xp_end"
    assert cleaned == "Have a great day."


def test_tag_with_captures():
    text = 'Confirmed. <route exit="xp_confirm" amount="500" date="2026-05-20" />'
    cleaned, tag = find_tag(text)
    assert tag.exit == "xp_confirm"
    assert tag.captures == {"amount": "500", "date": "2026-05-20"}
    assert cleaned == "Confirmed."


def test_tag_with_extra_whitespace_inside():
    text = 'OK. <route   exit="xp_X"   />'
    _, tag = find_tag(text)
    assert tag.exit == "xp_X"


def test_self_closing_tag_with_no_space_before_slash():
    text = 'OK. <route exit="xp_X"/>'
    _, tag = find_tag(text)
    assert tag.exit == "xp_X"


# --------------------------------------------------------------------------
# Malformed / ambiguous tags — fail closed (treated as no tag)
# --------------------------------------------------------------------------


def test_both_exit_and_interrupt_is_invalid():
    """Mutually exclusive. Caller treats as stay; we report tag-not-valid."""
    text = '<route exit="xp_X" interrupt="int_Y" />'
    _, tag = find_tag(text)
    assert tag is not None
    assert not tag.is_valid


def test_empty_route_tag_is_invalid():
    text = "Reply. <route />"
    _, tag = find_tag(text)
    assert tag is not None
    assert not tag.is_valid


def test_singled_quoted_values_are_not_accepted():
    """Format is strict — only double quotes. Single-quoted attrs come back
    empty, and the parser treats the tag as malformed (neither exit nor
    interrupt set)."""
    text = "<route exit='xp_X' />"
    _, tag = find_tag(text)
    assert tag is not None
    assert not tag.is_valid


def test_multiple_tags_first_wins():
    text = '<route exit="xp_A" /> ignored <route exit="xp_B" />'
    cleaned, tag = find_tag(text)
    assert tag.exit == "xp_A"
    # Subsequent tags are left in the cleaned text — they won't be parsed by
    # the dispatcher but a downstream consumer (events, logging) can spot the
    # duplicate.
    assert 'xp_B' in cleaned


def test_unrelated_xml_in_text_is_ignored():
    text = "I said <strong>yes</strong> to that."
    cleaned, tag = find_tag(text)
    assert tag is None
    assert cleaned == text


def test_closing_form_tag_is_not_accepted():
    """Spec uses self-closing tags only. A `<route>...</route>` form would
    be ambiguous about content semantics and is not supported."""
    text = "<route exit=\"xp_X\"></route>"
    _, tag = find_tag(text)
    assert tag is None


# --------------------------------------------------------------------------
# parse_tag_body — used by the streaming stripper after it buffers the body
# --------------------------------------------------------------------------


def test_parse_body_exit_only():
    tag = parse_tag_body('exit="xp_X"')
    assert tag == RouteTag(exit="xp_X", interrupt=None, captures=None)


def test_parse_body_interrupt_only():
    tag = parse_tag_body('interrupt="int_X"')
    assert tag == RouteTag(exit=None, interrupt="int_X", captures=None)


def test_parse_body_invalid_returns_none():
    assert parse_tag_body('exit="xp_X" interrupt="int_X"') is None
    assert parse_tag_body("") is None


# --------------------------------------------------------------------------
# to_llm_results — adapter to the legacy resolver shape
# --------------------------------------------------------------------------


def test_to_llm_results_exit_carries_captures():
    tag = RouteTag(exit="xp_X", interrupt=None, captures={"amount": "500"})
    assert to_llm_results(tag) == {
        "take_exit_path": {"exit_path_id": "xp_X", "amount": "500"}
    }


def test_to_llm_results_interrupt():
    tag = RouteTag(exit=None, interrupt="int_X", captures=None)
    assert to_llm_results(tag) == {
        "trigger_interrupt": {"interrupt_flow_id": "int_X"}
    }


def test_to_llm_results_invalid_returns_empty():
    tag = RouteTag(exit=None, interrupt=None, captures=None)
    assert to_llm_results(tag) == {}


# --------------------------------------------------------------------------
# `<think>...</think>` reserved sentinel
# --------------------------------------------------------------------------


def test_strip_think_blocks_removes_block():
    assert (
        strip_think_blocks("Hello <think>internal note</think> world")
        == "Hello  world"
    )


def test_strip_think_blocks_is_case_insensitive():
    assert strip_think_blocks("a <Think>x</Think> b") == "a  b"


def test_strip_think_blocks_handles_multiline_block():
    text = "Top\n<think>\nstep 1\nstep 2\n</think>\nBottom"
    assert strip_think_blocks(text) == "Top\n\nBottom"


def test_strip_think_blocks_leaves_other_tags_alone():
    """Only `<think>...</think>` is reserved. `<strong>`, `<VERIFICATION>`,
    etc. flow through untouched. See routing_protocol module docstring for
    the policy."""
    text = "<strong>hi</strong> <VERIFICATION>x</VERIFICATION>"
    assert strip_think_blocks(text) == text


def test_find_tag_strips_think_block_alongside_route_tag():
    """Both sentinels are removed; the reply text returns clean."""
    text = (
        'Sure thing. <think>route to coffee</think> '
        'On it. <route exit="xp_coffee" />'
    )
    cleaned, tag = find_tag(text)
    assert tag is not None and tag.exit == "xp_coffee"
    assert "<think" not in cleaned
    assert "<route" not in cleaned
    assert "Sure thing." in cleaned
    assert "On it." in cleaned
