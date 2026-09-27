"""Runtime message-shape rules the OpenAI API enforces.

These were all live 400s or crashes before they were tests.
"""

from __future__ import annotations

from athena.runtime.models import items_to_messages

USER = {"type": "message", "role": "user", "content": "go"}


def test_parallel_tool_calls_share_one_assistant_message():
    """Two calls in one turn must not become two assistant messages.

    The API rejects that with "An assistant message with 'tool_calls' must be
    followed by tool messages responding to each 'tool_call_id'".
    """
    messages = items_to_messages(
        [
            USER,
            {"type": "function_call", "call_id": "a", "name": "t1", "arguments": '{"x":1}'},
            {"type": "function_call", "call_id": "b", "name": "t2", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "ra"},
            {"type": "function_call_output", "call_id": "b", "output": "rb"},
        ]
    )
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool"]
    assert len(messages[1]["tool_calls"]) == 2
    assert {m["tool_call_id"] for m in messages[2:]} == {"a", "b"}


def test_unanswered_tool_call_gets_a_placeholder_result():
    """A run suspended at an approval gate leaves a call with no result."""
    messages = items_to_messages(
        [
            USER,
            {"type": "function_call", "call_id": "a", "name": "t1", "arguments": "{}"},
        ]
    )
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    assert messages[2]["tool_call_id"] == "a"


def test_every_tool_call_id_is_answered():
    """Invariant: no assistant tool_call may be left without a tool reply."""
    messages = items_to_messages(
        [
            USER,
            {"type": "function_call", "call_id": "a", "name": "t1", "arguments": "{}"},
            {"type": "function_call", "call_id": "b", "name": "t2", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "ra"},
        ]
    )
    called = {c["id"] for m in messages if m.get("tool_calls") for c in m["tool_calls"]}
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    assert called == answered


def test_consecutive_same_role_messages_merge():
    messages = items_to_messages(
        [USER, {"type": "message", "role": "user", "content": "again"}]
    )
    assert messages == [{"role": "user", "content": "go\nagain"}]


def test_transcript_must_open_with_a_user_turn():
    messages = items_to_messages(
        [{"type": "message", "role": "assistant", "content": "hi"}]
    )
    assert messages[0]["role"] == "user"
