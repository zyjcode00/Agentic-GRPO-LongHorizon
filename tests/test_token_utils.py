from __future__ import annotations

from src.utils.token_utils import get_grpo_masks, offset_alignment


class FakeQwen25Tokenizer:
    """Small deterministic tokenizer with Qwen2.5-like chat templates.

    It tokenizes each character into one token so token indices and decoded text
    are easy to reason about while preserving the semantics of
    apply_chat_template(add_generation_prompt=...).
    """

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        rendered = ""
        for message in messages:
            rendered += (
                f"<|im_start|>{message['role']}\n"
                f"{message['content']}"
                "<|im_end|>\n"
            )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        if not tokenize:
            return rendered
        return self._encode(rendered)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        encoded = {"input_ids": self._encode(text)}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return encoded

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)

    @staticmethod
    def _encode(text):
        return [ord(ch) for ch in text]


class DriftQwen25Tokenizer(FakeQwen25Tokenizer):
    """Tokenizer that simulates a one-token placeholder drift in prompt render."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = super().apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
        if tokenize and add_generation_prompt:
            assistant_header = self._encode("<|im_start|>assistant\n")
            # Insert an unexpected placeholder right before the generation prompt.
            return ids[:-len(assistant_header)] + [ord("¤")] + assistant_header
        return ids


def _decode_masked(tokenizer, input_ids, mask):
    return tokenizer.decode([token for token, keep in zip(input_ids, mask) if keep])


def test_get_grpo_masks_aligns_all_masks_to_qwen25_full_render():
    tokenizer = FakeQwen25Tokenizer()
    prompt_messages = [
        {"role": "system", "content": "You are an airline assistant."},
        {"role": "user", "content": "Find a flight from SFO to JFK."},
    ]
    response = (
        "<thought>Need to search flights first.</thought>\n"
        "Action: search_flights(origin='SFO', destination='JFK')\n"
        "Observation: AA100 is available.\n"
        "Final: I found AA100."
    )

    masks = get_grpo_masks(tokenizer, prompt_messages, response)
    full_render_ids = tokenizer.apply_chat_template(
        [*prompt_messages, {"role": "assistant", "content": response}],
        tokenize=True,
        add_generation_prompt=False,
    )

    assert masks.input_ids == full_render_ids
    assert len(masks.input_ids) == len(masks.attention_mask)
    assert len(masks.input_ids) == len(masks.response_mask)
    assert len(masks.input_ids) == len(masks.thought_mask)
    assert len(masks.input_ids) == len(masks.action_mask)
    assert set(masks.attention_mask) == {1}

    response_text = _decode_masked(tokenizer, masks.input_ids, masks.response_mask)
    assert response in response_text
    assert "Find a flight" not in response_text


def test_thought_and_action_masks_mark_only_response_subspans():
    tokenizer = FakeQwen25Tokenizer()
    response = (
        "<thought>Compare options and use the search tool.</thought>\n"
        "Action: search_flights(origin='SFO', destination='JFK')\n"
        "Observation: Found one itinerary.\n"
        "Final: The best option is AA100."
    )

    masks = get_grpo_masks(
        tokenizer,
        [{"role": "user", "content": "Need SFO to JFK"}],
        response,
    )

    thought_text = _decode_masked(tokenizer, masks.input_ids, masks.thought_mask)
    action_text = _decode_masked(tokenizer, masks.input_ids, masks.action_mask)

    assert thought_text == "Compare options and use the search tool."
    assert action_text == "search_flights(origin='SFO', destination='JFK')"
    assert "<thought>" not in thought_text
    assert "</thought>" not in thought_text
    assert "Action:" not in action_text
    assert "Observation:" not in action_text
    assert all(t <= r for t, r in zip(masks.thought_mask, masks.response_mask))
    assert all(a <= r for a, r in zip(masks.action_mask, masks.response_mask))


def test_get_grpo_masks_accepts_full_messages_argument():
    tokenizer = FakeQwen25Tokenizer()
    messages = [
        {"role": "user", "content": "Book a ticket"},
        {"role": "assistant", "content": "<thought>Need details.</thought>\nFinal: Please provide dates."},
    ]

    masks = get_grpo_masks(tokenizer, messages=messages)

    assert len(masks.input_ids) == len(masks.response_mask)
    assert _decode_masked(tokenizer, masks.input_ids, masks.thought_mask) == "Need details."


def test_offset_alignment_corrects_small_generation_prompt_drift():
    tokenizer = DriftQwen25Tokenizer()
    prompt_messages = [{"role": "user", "content": "Need help"}]
    response = "<thought>Plan.</thought>\nAction: help_user()\nFinal: Done."

    masks = get_grpo_masks(tokenizer, prompt_messages, response)
    response_text = _decode_masked(tokenizer, masks.input_ids, masks.response_mask)

    assert response in response_text
    assert "Need help" not in response_text
    assert _decode_masked(tokenizer, masks.input_ids, masks.thought_mask) == "Plan."
    assert _decode_masked(tokenizer, masks.input_ids, masks.action_mask) == "help_user()"


def test_offset_alignment_falls_back_to_common_prefix_when_no_window_matches():
    assert offset_alignment([1, 2, 3, 4], [1, 2, 9, 8]) == 2
