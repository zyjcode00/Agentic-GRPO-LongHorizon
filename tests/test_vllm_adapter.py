import asyncio
import time
from types import SimpleNamespace

from src.algorithms.grpo_trainer import GRPOTrainer
from src.models.vllm_adapter import (
    AsyncVLLMRollout,
    generate_and_score,
    get_current_env_session_id,
    get_current_tool_context,
)


class FakeAsyncEngine:
    """Async engine that verifies ContextVar isolation inside each request."""

    def __init__(self, delays=None):
        self.delays = dict(delays or {})
        self.seen_contexts = []

    async def generate(self, prompt, sampling_params, request_id):
        await asyncio.sleep(self.delays.get(prompt, 0.0))
        env_session_id = get_current_env_session_id()
        tool_context = get_current_tool_context()
        self.seen_contexts.append((prompt, env_session_id, dict(tool_context)))
        text = (
            f"Thought: handle {prompt}.\n"
            "Action: search_flights(origin='SFO', destination='JFK')\n"
            f"Final: done for {env_session_id}."
        )
        return SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    text=text,
                    logprobs=[-0.1, -0.2, -0.3],
                    token_ids=[101, 102, 103],
                )
            ]
        )


class StreamingFakeEngine:
    async def generate(self, prompt, sampling_params, request_id):
        async def iterator():
            yield {"outputs": [{"text": "partial", "token_ids": [1]}]}
            await asyncio.sleep(0)
            yield {"outputs": [{"text": f"final {prompt}", "logprobs": [-0.5], "token_ids": [1, 2]}]}

        return iterator()


def test_async_vllm_rollout_parallel_sampling_preserves_input_order_and_tps(capsys):
    engine = FakeAsyncEngine(delays={"slow": 0.04, "fast": 0.0})
    rollout = AsyncVLLMRollout(engine=engine, sampling_params={"max_tokens": 8}, print_tps=True)

    start = time.perf_counter()
    generations = asyncio.run(
        rollout.generate(
            ["slow", "fast"],
            contexts=[{"trace": "A"}, {"trace": "B"}],
            env_session_ids=["env-A", "env-B"],
        )
    )
    elapsed = time.perf_counter() - start
    captured = capsys.readouterr().out

    assert [generation.prompt for generation in generations] == ["slow", "fast"]
    assert [generation.env_session_id for generation in generations] == ["env-A", "env-B"]
    assert [generation.logprobs for generation in generations] == [[-0.1, -0.2, -0.3], [-0.1, -0.2, -0.3]]
    assert elapsed < 0.08
    assert "[vLLM Rollout]" in captured
    assert "TPS=" in captured
    assert rollout.last_metrics["tokens"] == 6.0
    assert rollout.last_metrics["tps"] > 0


def test_contextvars_isolate_concurrent_env_session_and_tool_context():
    engine = FakeAsyncEngine(delays={"A": 0.02, "B": 0.0, "C": 0.01})
    rollout = AsyncVLLMRollout(engine=engine, print_tps=False)

    generations = asyncio.run(
        rollout.generate(
            ["A", "B", "C"],
            contexts=[
                {"tool_context": {"task": "alpha", "allowed_tool": "search_flights"}},
                {"tool_context": {"task": "beta", "allowed_tool": "lookup_booking"}},
                {"tool_context": {"task": "gamma", "allowed_tool": "cancel_booking"}},
            ],
            env_session_ids=["session-A", "session-B", "session-C"],
        )
    )

    seen_by_prompt = {prompt: (session_id, context) for prompt, session_id, context in engine.seen_contexts}

    assert seen_by_prompt["A"] == ("session-A", {"task": "alpha", "allowed_tool": "search_flights"})
    assert seen_by_prompt["B"] == ("session-B", {"task": "beta", "allowed_tool": "lookup_booking"})
    assert seen_by_prompt["C"] == ("session-C", {"task": "gamma", "allowed_tool": "cancel_booking"})
    assert [generation.context["env_session_id"] for generation in generations] == [
        "session-A",
        "session-B",
        "session-C",
    ]


def test_generate_iter_accepts_vllm_style_async_iterator_output():
    rollout = AsyncVLLMRollout(engine=StreamingFakeEngine(), print_tps=False)

    async def collect():
        return [item async for item in rollout.generate_iter(["prompt"])]

    items = asyncio.run(collect())

    assert len(items) == 1
    index, generation = items[0]
    assert index == 0
    assert generation.response == "final prompt"
    assert generation.logprobs == [-0.5]
    assert generation.num_tokens == 2


def test_generate_and_score_ingests_rollouts_into_grpo_trainer(capsys):
    engine = FakeAsyncEngine()
    rollout = AsyncVLLMRollout(engine=engine, print_tps=True)
    trainer = GRPOTrainer(
        config={
            "reward_model": {
                "length_normalization": False,
                "tool_specs": {"search_flights": {"required_params": ["origin", "destination"]}},
            },
            "trainer": {"empty_cache_every": 0},
        }
    )

    batch = asyncio.run(
        generate_and_score(
            rollout,
            ["Need SFO-JFK", "Need SFO-JFK"],
            trainer=trainer,
            contexts=[{"customer": "A"}, {"customer": "B"}],
            group_ids=["same", "same"],
            env_session_ids=["tau-A", "tau-B"],
        )
    )
    captured = capsys.readouterr().out

    assert len(batch.generations) == 2
    assert len(batch.samples) == 2
    assert len(trainer.buffer) == 2
    assert batch.prompts == ["Need SFO-JFK", "Need SFO-JFK"]
    assert batch.group_ids == ["same", "same"]
    assert batch.contexts[0]["env_session_id"] == "tau-A"
    assert batch.contexts[1]["env_session_id"] == "tau-B"
    assert batch.total_tokens == 6
    assert batch.tps > 0
    assert trainer.buffer.sample_batch()[0].metadata["reward_breakdown"].raw_reward > 0
    assert "[vLLM Rollout]" in captured
    assert "[GRPO Diagnostics]" in captured
