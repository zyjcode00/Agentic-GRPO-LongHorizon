"""Final entrypoint for Agentic GRPO training on tau-bench style tasks.

Heuristic assumptions used by this script
-----------------------------------------
The research/training stack around tau-bench, veRL and vLLM can vary by cluster.
To keep the entrypoint runnable in both production and CI, the script follows a
few explicit heuristics:

* Configuration is loaded from ``configs/grpo_config.yaml`` and then optional
  CLI overrides are applied.
* If the config does not provide a model, Qwen2.5 is used by default via
  ``Qwen/Qwen2.5-7B-Instruct``.
* Rollout prompts are read from ``rollout.prompts`` / ``data.train_prompts`` if
  present. Otherwise a tiny tau-bench-airline style prompt set is used as a
  smoke-test fallback.
* Checkpoint saving is duck-typed: policy.save_pretrained, policy.save, or a
  JSON metadata checkpoint are used in that order.
* tau-bench unseen evaluation is duck-typed through an optional evaluator. When
  no evaluator is available, a deterministic reward-model based proxy is used so
  the checkpoint hook still reports an ``unseen_success_rate``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.grpo_trainer import GRPOTrainer, load_grpo_config
from src.models.vllm_adapter import AsyncVLLMRollout, RolloutBatch, generate_and_score

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

DEFAULT_TAU_AIRLINE_PROMPTS = [
    (
        "You are a tau-bench airline assistant. Customer PAX-007 wants one "
        "economy ticket from SFO to JFK on 2026-06-01. Use tools before confirming."
    ),
    (
        "You are a tau-bench airline assistant. Customer PAX-042 needs a flight "
        "from LAX to SEA tomorrow. Search available flights before booking."
    ),
]

DEFAULT_UNSEEN_PROMPTS = [
    (
        "[unseen] A customer wants to change an existing airline booking from "
        "BOS to DEN. Use the correct tools and provide a concise final answer."
    ),
    (
        "[unseen] A traveler asks for a refundable cabin option from ORD to SFO. "
        "Search first, then explain the result."
    ),
]


class AgenticGRPOEntrypoint:
    """Small orchestration layer around rollout generation and GRPO updates."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        rollout: AsyncVLLMRollout | None = None,
        trainer: GRPOTrainer | None = None,
        evaluator: Any | None = None,
    ) -> None:
        self.config = dict(config)
        self.rollout = rollout or build_rollout(self.config)
        self.trainer = trainer or GRPOTrainer(config=self.config)
        self.evaluator = evaluator

    async def train_loop(self) -> list[dict[str, float]]:
        train_cfg = dict(self.config.get("trainer", {}) or {})
        rollout_cfg = dict(self.config.get("rollout", {}) or {})
        checkpoint_cfg = dict(self.config.get("checkpoint", {}) or {})

        total_steps = int(train_cfg.get("total_steps", train_cfg.get("max_steps", 1)))
        checkpoint_every = int(checkpoint_cfg.get("save_every", train_cfg.get("checkpoint_every", 0)) or 0)
        output_dir = Path(checkpoint_cfg.get("output_dir", train_cfg.get("output_dir", "checkpoints/agentic_grpo")))
        prompts = load_prompts(self.config, split="train")
        batch_size = int(rollout_cfg.get("batch_size", train_cfg.get("train_batch_size", len(prompts))) or len(prompts))
        metrics_history: list[dict[str, float]] = []

        for step in range(1, total_steps + 1):
            batch_prompts = select_prompt_batch(prompts, step=step, batch_size=batch_size)
            group_ids = [f"train-{step}-{i}" for i in range(len(batch_prompts))]
            contexts = [build_context(prompt, split="train", step=step, index=i) for i, prompt in enumerate(batch_prompts)]
            env_session_ids = [f"train-step-{step}-env-{i}" for i in range(len(batch_prompts))]

            rollout_batch = await generate_and_score(
                self.rollout,
                batch_prompts,
                trainer=self.trainer,
                sampling_params=rollout_cfg.get("sampling_params"),
                contexts=contexts,
                group_ids=group_ids,
                env_session_ids=env_session_ids,
            )
            train_metrics = self.trainer.train_batch()
            step_metrics = summarize_iteration(
                step=step,
                rollout_batch=rollout_batch,
                trainer=self.trainer,
                train_metrics=train_metrics,
            )
            metrics_history.append(step_metrics)
            print_diagnostics(step_metrics)

            if checkpoint_every and step % checkpoint_every == 0:
                checkpoint_path = save_checkpoint(self.trainer, output_dir=output_dir, step=step, config=self.config)
                unseen_success_rate = await evaluate_tau_unseen(
                    self.rollout,
                    self.trainer,
                    self.config,
                    evaluator=self.evaluator,
                )
                print(
                    "[Checkpoint] "
                    f"step={step}, path={checkpoint_path}, "
                    f"tau_bench_unseen_success_rate={unseen_success_rate:.4f}"
                )

        return metrics_history


def build_rollout(config: Mapping[str, Any]) -> AsyncVLLMRollout:
    model_cfg = dict(config.get("model", {}) or {})
    rollout_cfg = dict(config.get("rollout", {}) or {})
    vllm_cfg = dict(config.get("vllm", {}) or {})

    model_name = (
        model_cfg.get("name")
        or model_cfg.get("model_name")
        or rollout_cfg.get("model")
        or vllm_cfg.get("model")
        or DEFAULT_MODEL
    )
    engine_args = dict(vllm_cfg.get("engine_args", {}) or rollout_cfg.get("engine_args", {}) or {})
    engine_args.setdefault("model", model_name)
    sampling_params = dict(vllm_cfg.get("sampling_params", {}) or rollout_cfg.get("sampling_params", {}) or {})
    if not sampling_params:
        sampling_params = {"temperature": 1.0, "max_tokens": 256}

    return AsyncVLLMRollout(
        model=model_name,
        engine_args=engine_args,
        sampling_params=sampling_params,
        max_concurrency=rollout_cfg.get("max_concurrency", vllm_cfg.get("max_concurrency")),
        print_tps=bool(rollout_cfg.get("print_tps", True)),
    )


def load_config(config_path: str | Path, overrides: Sequence[str] | None = None) -> dict[str, Any]:
    config = load_grpo_config(config_path=config_path)
    if overrides:
        override_conf = OmegaConf.from_dotlist(list(overrides))
        merged = OmegaConf.merge(OmegaConf.create(config), override_conf)
        config = OmegaConf.to_container(merged, resolve=True) or {}
    return dict(config)


def load_prompts(config: Mapping[str, Any], *, split: str) -> list[str]:
    data_cfg = dict(config.get("data", {}) or {})
    rollout_cfg = dict(config.get("rollout", {}) or {})
    eval_cfg = dict(config.get("eval", {}) or {})

    if split == "train":
        inline = rollout_cfg.get("prompts") or data_cfg.get("train_prompts") or data_cfg.get("prompts")
        path = data_cfg.get("train_path") or rollout_cfg.get("prompt_path")
        fallback = DEFAULT_TAU_AIRLINE_PROMPTS
    else:
        inline = eval_cfg.get("unseen_prompts") or data_cfg.get("unseen_prompts")
        path = eval_cfg.get("unseen_path") or data_cfg.get("unseen_path")
        fallback = DEFAULT_UNSEEN_PROMPTS

    if inline:
        return [str(item) for item in inline]
    if path:
        return read_prompt_file(Path(path))
    return list(fallback)


def read_prompt_file(path: Path) -> list[str]:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.suffix.lower() == ".jsonl":
        prompts = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            prompts.append(str(item.get("prompt") or item.get("instruction") or item))
        return prompts
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_prompt_batch(prompts: Sequence[str], *, step: int, batch_size: int) -> list[str]:
    if not prompts:
        raise ValueError("No prompts available for training")
    batch_size = max(1, int(batch_size))
    start = ((step - 1) * batch_size) % len(prompts)
    return [str(prompts[(start + offset) % len(prompts)]) for offset in range(batch_size)]


def build_context(prompt: str, *, split: str, step: int, index: int) -> dict[str, Any]:
    return {
        "split": split,
        "step": step,
        "index": index,
        "prompt": prompt,
        "tool_context": {
            "benchmark": "tau-bench",
            "subset": "unseen" if split == "unseen" else "train",
            "allowed_tools": ["search_flights", "book_reservation"],
        },
    }


def summarize_iteration(
    *,
    step: int,
    rollout_batch: RolloutBatch,
    trainer: GRPOTrainer,
    train_metrics: Mapping[str, Any],
) -> dict[str, float]:
    diagnostics = dict(getattr(trainer, "last_diagnostics", {}) or {})
    rewards = [sample.reward for sample in rollout_batch.samples if sample.reward is not None]
    overall_pass_at_1 = sum(1 for reward in rewards if float(reward) > 0.0) / len(rewards) if rewards else 0.0
    metrics = {
        "step": float(step),
        "overall_pass@1": float(overall_pass_at_1),
        "reward_variance": float(diagnostics.get("reward_variance", 0.0)),
        "average_length": float(diagnostics.get("average_length", 0.0)),
        "TPS": float(rollout_batch.tps or getattr(trainer, "last_tps", 0.0) or 0.0),
        "loss": float(train_metrics.get("loss", 0.0)),
        "optimizer_steps": float(train_metrics.get("optimizer_steps", 0.0)),
    }
    return metrics


def print_diagnostics(metrics: Mapping[str, float]) -> None:
    print(
        "[Train Diagnostics] "
        f"step={int(metrics['step'])}, "
        f"overall pass^1={metrics['overall_pass@1']:.4f}, "
        f"reward_variance={metrics['reward_variance']:.6f}, "
        f"average_length={metrics['average_length']:.2f}, "
        f"TPS={metrics['TPS']:.2f}, "
        f"loss={metrics['loss']:.6f}"
    )


def save_checkpoint(trainer: GRPOTrainer, *, output_dir: Path, step: int, config: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    policy = getattr(trainer, "policy", None)
    if policy is not None and hasattr(policy, "save_pretrained"):
        policy.save_pretrained(str(checkpoint_dir))
    elif policy is not None and hasattr(policy, "save"):
        policy.save(str(checkpoint_dir))
    else:
        metadata = {
            "step": step,
            "trainer_global_step": getattr(trainer, "global_step", None),
            "optimizer_steps": getattr(trainer, "optimizer_steps", None),
            "model": (config.get("model", {}) or {}).get("name", DEFAULT_MODEL),
        }
        (checkpoint_dir / "trainer_state.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return checkpoint_dir


async def evaluate_tau_unseen(
    rollout: AsyncVLLMRollout,
    trainer: GRPOTrainer,
    config: Mapping[str, Any],
    *,
    evaluator: Any | None = None,
) -> float:
    eval_cfg = dict(config.get("eval", {}) or {})
    prompts = load_prompts(config, split="unseen")
    max_cases = int(eval_cfg.get("max_unseen_cases", len(prompts)) or len(prompts))
    prompts = prompts[:max_cases]

    if evaluator is not None:
        result = evaluator.evaluate_unseen(prompts=prompts, rollout=rollout, trainer=trainer)
        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            result = await result
        if isinstance(result, Mapping):
            return float(result.get("success_rate", result.get("unseen_success_rate", 0.0)))
        return float(result)

    contexts = [build_context(prompt, split="unseen", step=0, index=i) for i, prompt in enumerate(prompts)]
    env_session_ids = [f"unseen-eval-env-{i}" for i in range(len(prompts))]
    generations = await rollout.generate(
        prompts,
        sampling_params=eval_cfg.get("sampling_params"),
        contexts=contexts,
        group_ids=[f"unseen-{i}" for i in range(len(prompts))],
        env_session_ids=env_session_ids,
    )
    successes = 0
    for generation in generations:
        reward = trainer.worker.score_response(generation.prompt, generation.response, context=generation.context)
        if reward.final_reward > float(eval_cfg.get("success_threshold", 0.0)):
            successes += 1
    return successes / len(generations) if generations else 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Agentic GRPO with Qwen2.5 + async vLLM rollouts")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "grpo_config.yaml"))
    parser.add_argument("--override", action="append", default=[], help="OmegaConf dotlist override, e.g. trainer.total_steps=10")
    parser.add_argument("--steps", type=int, default=None, help="Override trainer.total_steps")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Override checkpoint.save_every")
    return parser.parse_args(argv)


async def async_main(argv: Sequence[str] | None = None) -> list[dict[str, float]]:
    args = parse_args(argv)
    overrides = list(args.override or [])
    if args.steps is not None:
        overrides.append(f"trainer.total_steps={args.steps}")
    if args.checkpoint_every is not None:
        overrides.append(f"checkpoint.save_every={args.checkpoint_every}")
    config = load_config(args.config, overrides=overrides)
    entrypoint = AgenticGRPOEntrypoint(config)
    return await entrypoint.train_loop()


def main(argv: Sequence[str] | None = None) -> list[dict[str, float]]:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
