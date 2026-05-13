"""Token-level utilities for GRPO training.

This module implements the Render-Twice-Diff strategy used by chat-model
trainers to derive precise loss masks for assistant responses.  The algorithm
renders the prompt-only conversation with ``add_generation_prompt=True`` and the
full prompt+assistant conversation with ``add_generation_prompt=False``.  The
first token that belongs to the full render but not the prompt render is treated
as the assistant response start.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, MutableSequence, Sequence


@dataclass
class GRPOMasks:
    """Aligned token ids and GRPO loss masks.

    Every mask has exactly the same length as ``input_ids``:
    - ``response_mask`` marks assistant response tokens.
    - ``thought_mask`` marks tokens inside ``<thought>...</thought>`` spans.
    - ``action_mask`` marks tokens belonging to tool/action calls, currently
      ``<action>...</action>`` blocks and ``Action:`` lines/spans.
    """

    input_ids: list[int]
    attention_mask: list[int]
    response_mask: list[int]
    thought_mask: list[int]
    action_mask: list[int]

    def to_dict(self) -> dict[str, list[int]]:
        """Return a plain dictionary compatible with tensor collators."""

        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "response_mask": self.response_mask,
            "thought_mask": self.thought_mask,
            "action_mask": self.action_mask,
        }


def get_grpo_masks(
    tokenizer: Any,
    prompt_messages: Sequence[Mapping[str, str]] | None = None,
    response: str | None = None,
    *,
    messages: Sequence[Mapping[str, str]] | None = None,
) -> GRPOMasks:
    """Build response/thought/action masks using Render-Twice-Diff.

    Args:
        tokenizer: A HuggingFace-style tokenizer exposing
            ``apply_chat_template``. Qwen2.5 Instruct tokenizers are supported.
        prompt_messages: Conversation messages before assistant generation.
        response: Assistant response content. Required when ``messages`` is not
            supplied.
        messages: Optional full conversation whose final message is the
            assistant response. This is a convenience alternative to passing
            ``prompt_messages`` and ``response`` separately.

    Returns:
        ``GRPOMasks`` with all fields aligned to ``input_ids``.
    """

    prompt_messages_list, response_text, full_messages = _normalise_inputs(
        prompt_messages=prompt_messages,
        response=response,
        messages=messages,
    )

    prompt_ids = _render_chat(tokenizer, prompt_messages_list, add_generation_prompt=True)
    full_ids = _render_chat(tokenizer, full_messages, add_generation_prompt=False)
    response_start = offset_alignment(prompt_ids, full_ids)

    input_ids = list(full_ids)
    attention_mask = [1] * len(input_ids)
    response_mask = [0] * len(input_ids)
    thought_mask = [0] * len(input_ids)
    action_mask = [0] * len(input_ids)

    response_end = len(input_ids)
    for idx in range(response_start, response_end):
        response_mask[idx] = 1

    response_token_offsets = _response_token_char_offsets(tokenizer, response_text)
    for char_start, char_end in _find_thought_spans(response_text):
        _fill_mask_for_char_span(
            thought_mask,
            response_start,
            response_token_offsets,
            char_start,
            char_end,
            max_len=len(input_ids),
        )

    for char_start, char_end in _find_action_spans(response_text):
        _fill_mask_for_char_span(
            action_mask,
            response_start,
            response_token_offsets,
            char_start,
            char_end,
            max_len=len(input_ids),
        )

    _intersect_in_place(thought_mask, response_mask)
    _intersect_in_place(action_mask, response_mask)

    return GRPOMasks(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        thought_mask=thought_mask,
        action_mask=action_mask,
    )


def offset_alignment(prompt_ids: Sequence[int], full_ids: Sequence[int]) -> int:
    """Return response start, correcting small chat-template prefix drift.

    The happy path is a longest common prefix. Some tokenizers/chat templates can
    insert or omit placeholder tokens around the assistant generation prompt, so
    this function falls back to finding the longest suffix of ``prompt_ids`` that
    appears as a prefix-ending window in ``full_ids``. The returned offset is
    clamped into the valid full sequence range.
    """

    common = 0
    max_common = min(len(prompt_ids), len(full_ids))
    while common < max_common and prompt_ids[common] == full_ids[common]:
        common += 1

    if common == len(prompt_ids) or common == len(full_ids):
        return min(common, len(full_ids))

    best_end = common
    best_len = 0
    max_window = min(len(prompt_ids), len(full_ids), 64)
    for window in range(max_window, 0, -1):
        suffix = list(prompt_ids[-window:])
        search_start = max(0, len(prompt_ids) - window - 16)
        search_end = min(len(full_ids) - window + 1, len(prompt_ids) + 17)
        for pos in range(search_start, max(search_start, search_end)):
            if list(full_ids[pos : pos + window]) == suffix:
                end = pos + window
                if window > best_len or (window == best_len and abs(end - len(prompt_ids)) < abs(best_end - len(prompt_ids))):
                    best_len = window
                    best_end = end
        if best_len:
            break

    if best_len:
        return min(best_end, len(full_ids))

    # Last-resort conservative behaviour: use the common-prefix boundary rather
    # than raising, so the trainer can still produce aligned masks.
    return min(common, len(full_ids))


def _normalise_inputs(
    *,
    prompt_messages: Sequence[Mapping[str, str]] | None,
    response: str | None,
    messages: Sequence[Mapping[str, str]] | None,
) -> tuple[list[dict[str, str]], str, list[dict[str, str]]]:
    if messages is not None:
        if not messages:
            raise ValueError("messages must not be empty")
        full_messages = [_copy_message(message) for message in messages]
        final = full_messages[-1]
        if final.get("role") != "assistant":
            raise ValueError("messages must end with an assistant response")
        response_text = final.get("content", "") if response is None else response
        return full_messages[:-1], response_text, full_messages

    if prompt_messages is None:
        raise ValueError("prompt_messages is required when messages is not supplied")
    if response is None:
        raise ValueError("response is required when messages is not supplied")

    prompt = [_copy_message(message) for message in prompt_messages]
    full = [*prompt, {"role": "assistant", "content": response}]
    return prompt, response, full


def _copy_message(message: Mapping[str, str]) -> dict[str, str]:
    return {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}


def _render_chat(tokenizer: Any, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered.get("input_ids", [])
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(token_id) for token_id in rendered]


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    tokenized = tokenizer(text, add_special_tokens=False)
    ids = tokenized.get("input_ids", tokenized) if isinstance(tokenized, Mapping) else tokenized
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _response_token_char_offsets(tokenizer: Any, response: str) -> list[tuple[int, int]]:
    """Approximate character offsets for response tokens.

    Prefer fast-tokenizer offset mappings. If unavailable, fall back to decoding
    each token piece and accumulating character lengths.
    """

    try:
        encoded = tokenizer(
            response,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping") if isinstance(encoded, Mapping) else None
        if offsets is not None:
            if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], tuple):
                offsets = offsets[0]
            return [(int(start), int(end)) for start, end in offsets]
    except (NotImplementedError, TypeError, ValueError):
        pass

    token_ids = _tokenize_text(tokenizer, response)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token_id in token_ids:
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
        start = cursor
        end = min(len(response), start + len(piece))
        offsets.append((start, end))
        cursor = end
    return offsets


def _find_thought_spans(response: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"<thought\b[^>]*>(.*?)</thought>", response, flags=re.IGNORECASE | re.DOTALL):
        spans.append((match.start(1), match.end(1)))
    return spans


def _find_action_spans(response: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"<action\b[^>]*>(.*?)</action>", response, flags=re.IGNORECASE | re.DOTALL):
        spans.append((match.start(1), match.end(1)))

    action_pattern = re.compile(
        r"(?im)^\s*Action\s*:\s*(.*?)(?=^\s*(?:Observation|Final|Thought|Action)\s*:|\Z)",
        flags=re.DOTALL,
    )
    for match in action_pattern.finditer(response):
        start, end = match.start(1), match.end(1)
        while end > start and response[end - 1] in "\r\n":
            end -= 1
        spans.append((start, end))
    return _merge_spans(spans)


def _merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((start, end) for start, end in spans if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _fill_mask_for_char_span(
    mask: MutableSequence[int],
    response_start: int,
    token_offsets: Sequence[tuple[int, int]],
    char_start: int,
    char_end: int,
    *,
    max_len: int,
) -> None:
    for rel_token_idx, (tok_start, tok_end) in enumerate(token_offsets):
        if tok_end <= char_start or tok_start >= char_end:
            continue
        abs_token_idx = response_start + rel_token_idx
        if 0 <= abs_token_idx < max_len:
            mask[abs_token_idx] = 1


def _intersect_in_place(mask: MutableSequence[int], gate: Sequence[int]) -> None:
    for idx in range(min(len(mask), len(gate))):
        mask[idx] = 1 if mask[idx] and gate[idx] else 0
