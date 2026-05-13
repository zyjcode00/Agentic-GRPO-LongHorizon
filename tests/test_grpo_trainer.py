import math
from pathlib import Path

from src.algorithms.grpo_trainer import (
    DistributedGRPOBuffer,
    GRPOTrainer,
    GRPOWorker,
    build_reward_model_from_config,
    compute_advantages_distributed,
    load_grpo_config,
)
from src.algorithms.reward_model import GRPORewardModel, ToolSpec


GOOD_RESPONSE = """
Thought: The user needs SFO to JFK, so I should search flights.
Action: search_flights(origin='SFO', destination='JFK', date='2026-06-01')
Observation: Flight AA100 is available.
Final: Book AA100 if the passenger confirms.
"""

BAD_RESPONSE = """
Thought: I will use a random unsupported API.
Action: delete_booking(id='oops')
Final: Done.
"""


def test_worker_scores_vllm_text_and_returns_reward_breakdown():
    worker = GRPOWorker(
        GRPORewardModel(
            tool_specs={
                "search_flights": ToolSpec(
                    required_params={"origin", "destination"},
                    optional_params={"date"},
                )
            },
            length_normalization=False,
        )
    )

    result = worker.score_response("Need SFO to JFK", GOOD_RESPONSE)

    assert result.raw_reward > 0
    assert result.final_reward == result.raw_reward
    assert any("valid_tool_call" in step.reasons for step in result.step_rewards)


def test_compute_advantages_distributed_uses_gathered_global_rewards():
    model = GRPORewardModel(length_normalization=False)

    def fake_gather(local_payload):
        return [
            {"rewards": [1.0, 2.0], "group_ids": ["p", "p"]},
            local_payload,
        ]

    local_advantages = compute_advantages_distributed(
        [3.0, 4.0],
        group_ids=["p", "p"],
        reward_model=model,
        gather_fn=fake_gather,
    )
    all_advantages = model.compute_group_advantages([1.0, 2.0, 3.0, 4.0], group_ids=["p"] * 4)

    assert local_advantages == all_advantages[2:]
    assert math.isclose(sum(all_advantages), 0.0, abs_tol=1e-7)
    variance = sum(a * a for a in all_advantages) / len(all_advantages)
    assert math.isclose(variance, 1.0, rel_tol=1e-6)


def test_distributed_buffer_stores_prompt_response_logprobs_advantages():
    buffer = DistributedGRPOBuffer()

    buffer.add("prompt", "response", [-0.1, -0.2], 1.25, group_id="prompt", reward=0.8)
    batch = buffer.sample_batch(clear=True)

    assert len(batch) == 1
    assert batch[0].prompt == "prompt"
    assert batch[0].response == "response"
    assert batch[0].logprobs == [-0.1, -0.2]
    assert batch[0].advantage == 1.25
    assert batch[0].group_id == "prompt"
    assert len(buffer) == 0


def test_reward_model_hyperparameters_load_from_yaml(tmp_path: Path):
    config_path = tmp_path / "grpo_config.yaml"
    config_path.write_text(
        """
reward_model:
  correct_tool_reward: 2.5
  invalid_tool_penalty: -1.5
  length_normalization: false
  tool_specs:
    search_flights:
      required_params: [origin, destination]
trainer:
  gradient_accumulation_steps: 3
""".strip(),
        encoding="utf-8",
    )

    config = load_grpo_config(config_path)
    model = build_reward_model_from_config(config_path=config_path)

    assert config["reward_model"]["correct_tool_reward"] == 2.5
    assert model.correct_tool_reward == 2.5
    assert model.invalid_tool_penalty == -1.5
    assert "search_flights" in model.tool_specs
    result = model.compute_reward("Action: search_flights(origin='SFO', destination='JFK')")
    assert result.raw_reward == 1.0
    assert result.step_rewards[0].reasons == ("valid_tool_call",)


def test_trainer_ingests_rollouts_logs_diagnostics_and_accumulates_gradients(capsys):
    class DummyOptimizer:
        def __init__(self):
            self.steps = 0
            self.zeroes = 0

        def step(self):
            self.steps += 1

        def zero_grad(self):
            self.zeroes += 1

    optimizer = DummyOptimizer()
    seen_micro_batches = []

    def loss_fn(policy, micro_batch):
        seen_micro_batches.append(len(micro_batch))
        return sum(sample.advantage for sample in micro_batch) + 1.0

    trainer = GRPOTrainer(
        optimizer=optimizer,
        config={
            "reward_model": {
                "length_normalization": False,
                "tool_specs": {"search_flights": {"required_params": ["origin", "destination"]}},
            },
            "trainer": {"gradient_accumulation_steps": 2, "empty_cache_every": 0},
        },
        loss_fn=loss_fn,
    )

    samples = trainer.ingest_rollouts(
        ["Need SFO-JFK", "Need SFO-JFK", "Need SFO-JFK", "Need SFO-JFK"],
        [GOOD_RESPONSE, BAD_RESPONSE, GOOD_RESPONSE, BAD_RESPONSE],
        [[-0.1], [-0.2], [-0.3], [-0.4]],
        group_ids=["same"] * 4,
    )
    captured = capsys.readouterr().out

    assert len(samples) == 4
    assert "[GRPO Diagnostics]" in captured
    assert trainer.last_diagnostics["reward_variance"] > 0
    assert trainer.last_diagnostics["average_length"] > 0

    metrics = trainer.train_batch()

    assert seen_micro_batches == [2, 2]
    assert optimizer.steps == 1
    assert optimizer.zeroes >= 2
    assert metrics["optimizer_steps"] == 1.0
