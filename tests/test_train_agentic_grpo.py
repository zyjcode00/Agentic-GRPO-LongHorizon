import asyncio
import json
from types import SimpleNamespace

from scripts.train_agentic_grpo import (
    AgenticGRPOEntrypoint,
    build_rollout,
    evaluate_tau_unseen,
    load_config,
)
from src.algorithms.grpo_trainer import GRPOTrainer
from src.models.vllm_adapter import AsyncVLLMRollout


class FakeAsyncEngine:
    def __init__(self):
        self.calls = []

    async def generate(self, prompt, sampling_params, request_id):
        self.calls.append((prompt, sampling_params, request_id))
        return SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    text=(
                        "Thought: search before booking.\n"
                        "Action: search_flights(origin='SFO', destination='JFK')\n"
                        "Final: valid itinerary found."
                    ),
                    logprobs=[-0.1, -0.2],
                    token_ids=[1, 2, 3, 4],
                )
            ]
        )


class RecordingTrainer(GRPOTrainer):
    def __init__(self, config):
        super().__init__(config=config)
        self.train_batch_calls = 0

    def train_batch(self, batch=None):
        self.train_batch_calls += 1
        metrics = super().train_batch(batch=batch)
        metrics["optimizer_steps"] = float(self.train_batch_calls)
        return metrics


class FakePolicy:
    def __init__(self):
        self.saved_paths = []

    def save_pretrained(self, path):
        self.saved_paths.append(path)
        with open(f"{path}/policy.txt", "w", encoding="utf-8") as handle:
            handle.write("saved")


class FakeEvaluator:
    def __init__(self):
        self.calls = []

    async def evaluate_unseen(self, *, prompts, rollout, trainer):
        self.calls.append((list(prompts), rollout, trainer))
        return {"success_rate": 0.55}


def _config(tmp_path):
    return {
        "reward_model": {
            "length_normalization": False,
            "tool_specs": {"search_flights": {"required_params": ["origin", "destination"]}},
        },
        "trainer": {"total_steps": 2, "train_batch_size": 2, "empty_cache_every": 0},
        "rollout": {
            "batch_size": 2,
            "print_tps": False,
            "prompts": ["case A", "case B", "case C"],
            "sampling_params": {"temperature": 0.7, "max_tokens": 16},
        },
        "checkpoint": {"save_every": 1, "output_dir": str(tmp_path / "ckpts")},
        "eval": {"unseen_prompts": ["unseen A", "unseen B"], "max_unseen_cases": 1},
        "model": {"name": "Qwen/Qwen2.5-7B-Instruct"},
    }


def test_load_config_merges_yaml_and_dotlist_overrides(tmp_path):
    config_path = tmp_path / "grpo_config.yaml"
    config_path.write_text(
        "trainer:\n  total_steps: 1\nrollout:\n  batch_size: 2\nmodel:\n  name: old-model\n",
        encoding="utf-8",
    )

    config = load_config(config_path, overrides=["trainer.total_steps=3", "model.name=Qwen/Qwen2.5-7B-Instruct"])

    assert config["trainer"]["total_steps"] == 3
    assert config["rollout"]["batch_size"] == 2
    assert config["model"]["name"] == "Qwen/Qwen2.5-7B-Instruct"


def test_build_rollout_uses_qwen25_default_and_configured_sampling(monkeypatch):
    captured = {}

    def fake_init(self, model=None, **kwargs):
        captured["model"] = model
        captured.update(kwargs)

    monkeypatch.setattr(AsyncVLLMRollout, "__init__", fake_init)

    build_rollout({"rollout": {"sampling_params": {"max_tokens": 32}, "max_concurrency": 4}})

    assert captured["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert captured["engine_args"]["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert captured["sampling_params"] == {"max_tokens": 32}
    assert captured["max_concurrency"] == 4


def test_train_loop_generates_scores_trains_prints_checkpoints_and_evaluates_unseen(tmp_path, capsys):
    config = _config(tmp_path)
    engine = FakeAsyncEngine()
    rollout = AsyncVLLMRollout(engine=engine, sampling_params={"max_tokens": 8}, print_tps=False)
    trainer = RecordingTrainer(config=config)
    trainer.policy = FakePolicy()
    evaluator = FakeEvaluator()
    entrypoint = AgenticGRPOEntrypoint(config, rollout=rollout, trainer=trainer, evaluator=evaluator)

    history = asyncio.run(entrypoint.train_loop())
    output = capsys.readouterr().out

    assert trainer.train_batch_calls == 2
    assert len(history) == 2
    assert all(metrics["overall_pass@1"] == 1.0 for metrics in history)
    assert all(metrics["average_length"] > 0 for metrics in history)
    assert all(metrics["TPS"] > 0 for metrics in history)
    assert "overall pass^1" in output
    assert "reward_variance" in output
    assert "average_length" in output
    assert "TPS" in output
    assert "tau_bench_unseen_success_rate=0.5500" in output
    assert len(evaluator.calls) == 2
    assert len(trainer.policy.saved_paths) == 2
    assert (tmp_path / "ckpts" / "step_000001" / "policy.txt").exists()
    assert (tmp_path / "ckpts" / "step_000002" / "policy.txt").exists()


def test_checkpoint_without_policy_writes_metadata_and_proxy_unseen_eval(tmp_path):
    config = _config(tmp_path)
    config["trainer"]["total_steps"] = 1
    config["checkpoint"]["save_every"] = 1
    engine = FakeAsyncEngine()
    rollout = AsyncVLLMRollout(engine=engine, print_tps=False)
    trainer = RecordingTrainer(config=config)
    entrypoint = AgenticGRPOEntrypoint(config, rollout=rollout, trainer=trainer)

    asyncio.run(entrypoint.train_loop())

    state_path = tmp_path / "ckpts" / "step_000001" / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["step"] == 1
    assert state["model"] == "Qwen/Qwen2.5-7B-Instruct"


def test_evaluate_tau_unseen_accepts_sync_evaluator_result(tmp_path):
    class SyncEvaluator:
        def evaluate_unseen(self, *, prompts, rollout, trainer):
            return {"unseen_success_rate": 0.75}

    config = _config(tmp_path)
    rollout = AsyncVLLMRollout(engine=FakeAsyncEngine(), print_tps=False)
    trainer = RecordingTrainer(config=config)

    rate = asyncio.run(evaluate_tau_unseen(rollout, trainer, config, evaluator=SyncEvaluator()))

    assert rate == 0.75
