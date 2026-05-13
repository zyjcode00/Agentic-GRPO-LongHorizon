"""Integration tests for tau-bench style airline trajectories.

The tests intentionally use realistic rollout text with ``<thought>``,
``Action:``, ``Observation:`` and ``Final:`` markers so that the lightweight
reward model is exercised in the same format used by agent-environment traces.
"""

from __future__ import annotations

import math

from src.algorithms.reward_model import GRPORewardModel, RewardBreakdown, ToolSpec


AIRLINE_TOOL_SPECS = {
    "search_flights": ToolSpec(
        required_params={"origin", "destination", "date"},
        optional_params={"passengers", "cabin"},
        validators={
            "origin": lambda value: isinstance(value, str) and len(value) == 3,
            "destination": lambda value: isinstance(value, str) and len(value) == 3,
            "date": lambda value: isinstance(value, str) and len(value.split("-")) == 3,
            "passengers": lambda value: isinstance(value, int) and value > 0,
        },
    ),
    "book_reservation": ToolSpec(
        required_params={"flight_id", "passenger_id"},
        optional_params={"seat"},
        validators={
            "flight_id": lambda value: isinstance(value, str) and value.startswith("UA"),
            "passenger_id": lambda value: isinstance(value, str) and value.startswith("PAX"),
        },
    ),
}


PROMPT = (
    "You are a tau-bench airline assistant. Customer PAX-007 wants one economy "
    "ticket from SFO to JFK on 2026-06-01. Use tools before confirming."
)


SHORT_TAU_AIRLINE_TRAJECTORY = """
<thought>
Thought: Because the customer PAX-007 asked for SFO to JFK on 2026-06-01, I should search flights first.
</thought>
Action: search_flights(origin='SFO', destination='JFK', date='2026-06-01', passengers=1, cabin='economy')
Observation: search_flights returned UA100 from SFO to JFK on 2026-06-01 with one economy seat available.
<thought>
Thought: Based on the observation, UA100 is available for passenger PAX-007, so I can book it.
</thought>
Action: book_reservation(flight_id='UA100', passenger_id='PAX-007', seat='12A')
Observation: Reservation R-9001 confirmed for PAX-007 on UA100, seat 12A.
Final: Based on reservation R-9001, the customer is booked from SFO to JFK on 2026-06-01.
"""


LONG_TAU_AIRLINE_TRAJECTORY = """
<thought>
Thought: Because the customer PAX-007 asked for SFO to JFK on 2026-06-01, I should search flights first. I will carefully restate the airline task, the passenger identifier, the route, the requested date, the cabin, and the need to avoid unsupported conclusions before using a tool. This verbose reasoning is intentionally much longer than necessary, and it repeats harmless planning details so that the LATA square-root length normalisation has a noticeably larger denominator while the semantic steps stay valid.
</thought>
Action: search_flights(origin='SFO', destination='JFK', date='2026-06-01', passengers=1, cabin='economy')
Observation: search_flights returned UA100 from SFO to JFK on 2026-06-01 with one economy seat available.
<thought>
Thought: Based on the observation, UA100 is available for passenger PAX-007, so I can book it. I will again spell out that the observation explicitly contains the flight id, route, date, and seat availability. This additional wording should not create more process reward because it remains inside the same thought step, but it should increase the measured trajectory length and therefore compress the final reward under LATA.
</thought>
Action: book_reservation(flight_id='UA100', passenger_id='PAX-007', seat='12A')
Observation: Reservation R-9001 confirmed for PAX-007 on UA100, seat 12A.
Final: Based on reservation R-9001, the customer is booked from SFO to JFK on 2026-06-01.
"""


INVALID_TOOL_TRAJECTORY = """
<thought>
Thought: The customer wants a booking, but I will call an unrelated tool without enough airline arguments.
</thought>
Action: cancel_reservation(passenger_id='PAX-007')
Observation: The tool call failed because cancel_reservation is unavailable in this task.
Final: I cannot confirm a valid booking from this failed tool call.
"""


MISSING_ARGUMENT_TRAJECTORY = """
<thought>
Thought: Because the customer said SFO and JFK, I need a flight search, but I forget the required date.
</thought>
Action: search_flights(origin='SFO', destination='JFK', passengers=1)
Observation: search_flights rejected the call because date is required.
Final: I still need the missing date before confirming the itinerary.
"""


def _build_model(length_normalization: bool = True) -> GRPORewardModel:
    return GRPORewardModel(
        tool_specs=AIRLINE_TOOL_SPECS,
        length_normalization=length_normalization,
    )


def test_tau_bench_airline_trajectory_scores_with_prm_lite_and_lata() -> None:
    model = _build_model(length_normalization=True)

    result = model.compute_reward(
        SHORT_TAU_AIRLINE_TRAJECTORY,
        prompt=PROMPT,
        outcome_reward=1.0,
    )

    assert isinstance(result, RewardBreakdown)
    assert result.raw_reward > 1.0
    assert result.final_reward == result.raw_reward / math.sqrt(result.length)

    reasons = [reason for step in result.step_rewards for reason in step.reasons]
    assert reasons.count("valid_tool_call") == 2
    assert "observation_integrated" in reasons
    assert "grounded_final_answer" in reasons


def test_lata_compresses_longer_tau_trajectory_with_same_valid_actions() -> None:
    raw_model = _build_model(length_normalization=False)
    lata_model = _build_model(length_normalization=True)

    short_raw = raw_model.compute_reward(SHORT_TAU_AIRLINE_TRAJECTORY, prompt=PROMPT, outcome_reward=1.0)
    long_raw = raw_model.compute_reward(LONG_TAU_AIRLINE_TRAJECTORY, prompt=PROMPT, outcome_reward=1.0)
    short_lata = lata_model.compute_reward(SHORT_TAU_AIRLINE_TRAJECTORY, prompt=PROMPT, outcome_reward=1.0)
    long_lata = lata_model.compute_reward(LONG_TAU_AIRLINE_TRAJECTORY, prompt=PROMPT, outcome_reward=1.0)

    # The two trajectories contain the same valid actions, observations and final answer.
    assert math.isclose(short_raw.raw_reward, long_raw.raw_reward, rel_tol=1e-12)
    assert long_lata.length > short_lata.length
    assert long_lata.final_reward < short_lata.final_reward
    assert math.isclose(long_lata.final_reward, long_lata.raw_reward / math.sqrt(long_lata.length), rel_tol=1e-12)


def test_four_sample_tau_group_advantages_are_normalized_and_printed() -> None:
    model = _build_model(length_normalization=True)
    trajectories = [
        SHORT_TAU_AIRLINE_TRAJECTORY,
        LONG_TAU_AIRLINE_TRAJECTORY,
        INVALID_TOOL_TRAJECTORY,
        MISSING_ARGUMENT_TRAJECTORY,
    ]
    rewards = [model.compute_reward(text, prompt=PROMPT, outcome_reward=1.0) for text in trajectories]
    group_advantages = model.compute_group_advantages(rewards, group_ids=["airline-case-1"] * 4)

    print("tau_group_final_rewards=", [round(reward.final_reward, 6) for reward in rewards])
    print("tau_group_advantages=", [round(advantage, 6) for advantage in group_advantages])

    assert len(group_advantages) == 4
    assert math.isclose(sum(group_advantages), 0.0, abs_tol=1e-7)
    assert math.isclose(
        sum(advantage**2 for advantage in group_advantages) / len(group_advantages),
        1.0,
        rel_tol=1e-6,
    )

    # The valid concise trajectory should be above the group mean, while invalid
    # or incomplete tool calls should be below it.
    assert group_advantages[0] > 0
    assert group_advantages[2] < 0
    assert group_advantages[3] < 0
    assert rewards[0].final_reward > rewards[1].final_reward > rewards[2].final_reward
