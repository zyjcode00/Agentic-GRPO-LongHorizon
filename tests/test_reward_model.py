import math

from src.algorithms.reward_model import (
    GRPORewardModel,
    RewardBreakdown,
    ToolSpec,
    apply_lata,
    compute_group_advantages,
)


def test_prm_lite_rewards_valid_tool_and_grounded_reasoning_more_than_invalid_call():
    model = GRPORewardModel(
        tool_specs={
            "search_flight": ToolSpec(
                required_params={"origin", "destination"},
                optional_params={"date"},
                validators={
                    "origin": lambda value: isinstance(value, str) and len(value) == 3,
                    "destination": lambda value: isinstance(value, str) and len(value) == 3,
                },
            )
        },
        length_normalization=False,
    )

    prompt = "User needs a flight from SFO to JFK on 2026-06-01."
    good_output = """
Thought: Because the user said SFO and JFK, I should search that route.
Action: search_flight(origin='SFO', destination='JFK', date='2026-06-01')
Observation: Found flight AA100 from SFO to JFK.
Final: Based on AA100, provide the itinerary.
"""
    bad_output = """
Thought: I will randomly call a flight API.
Action: search_flight(origin='San Francisco', extra='noise')
Final: Done.
"""

    good = model.compute_reward(good_output, prompt=prompt)
    bad = model.compute_reward(bad_output, prompt=prompt)

    assert good.raw_reward > bad.raw_reward
    assert any("valid_tool_call" in step.reasons for step in good.step_rewards)
    assert any(
        reason.startswith("missing_required_params") or reason.startswith("unknown_params") or reason.startswith("invalid_param")
        for step in bad.step_rewards
        for reason in step.reasons
    )


def test_prm_lite_rejects_unsupported_tool_and_ungrounded_number_claim():
    model = GRPORewardModel(
        tool_specs={"lookup_order": ToolSpec(required_params={"order_id"})},
        length_normalization=False,
    )
    output = """
Thought: The refund should be 999 dollars.
Action: delete_order(order_id='A1')
"""

    result = model.compute_reward(output, prompt="Order A1 has refund amount 10 dollars.")

    reasons = [reason for step in result.step_rewards for reason in step.reasons]
    assert "unsupported_reasoning" in reasons
    assert "unsupported_tool" in reasons
    assert result.raw_reward < 0


def test_lata_penalizes_verbose_output_with_same_raw_reward():
    short = apply_lata(4.0, length=4)
    long = apply_lata(4.0, length=100)

    assert short == 2.0
    assert long == 0.4
    assert short > long


def test_compute_reward_applies_lata_to_raw_reward():
    model = GRPORewardModel(length_normalization=True)
    output = "Thought: because prompt says Paris, answer Paris."

    result = model.compute_reward(output, prompt="The destination is Paris.", outcome_reward=1.0)

    assert isinstance(result, RewardBreakdown)
    assert result.final_reward == result.raw_reward / math.sqrt(result.length)


def test_group_advantage_normalizes_each_prompt_group_independently():
    advantages = compute_group_advantages(
        [1.0, 2.0, 3.0, 10.0, 20.0],
        group_ids=["prompt-a", "prompt-a", "prompt-a", "prompt-b", "prompt-b"],
    )

    assert math.isclose(advantages[0], -1.224744856391589, rel_tol=1e-12)
    assert math.isclose(advantages[1], 0.0, abs_tol=1e-12)
    assert math.isclose(advantages[2], 1.224744856391589, rel_tol=1e-12)
    assert advantages[3] < 0
    assert advantages[4] > 0
    assert math.isclose(sum(advantages[:3]), 0.0, abs_tol=1e-7)
    assert math.isclose(sum(advantages[3:]), 0.0, abs_tol=1e-7)


def test_group_advantage_zero_variance_group_returns_zeroes():
    assert compute_group_advantages([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]
