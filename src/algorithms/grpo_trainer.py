"""veRL/Ray-style GRPO training utilities.

中文导读：本文件是项目中“训练器 / rollout 样本缓冲区 / advantage 计算”的核心实现。
它负责把 vLLM 生成的回答转换成 GRPO 可训练样本，并提供奖励打分、分组优势归一化、
梯度累积、训练诊断和 CUDA cache 清理等训练入口脚本需要调用的能力。


This module intentionally keeps the core training loop lightweight and easy to
unit test while exposing the integration points needed by an industrial GRPO
pipeline:

* ``GRPOWorker`` owns a ``GRPORewardModel`` and scores vLLM rollout text.
* ``compute_advantages_distributed`` gathers rewards from Ray workers and
  normalises them globally with ``GRPORewardModel.compute_group_advantages``.
* ``DistributedGRPOBuffer`` stores prompt/response/logprob/advantage records.
* ``GRPOTrainer`` implements diagnostics, cache hooks and gradient accumulation.

The implementation does not require Ray, veRL or CUDA at import time. When Ray is
not available, the distributed advantage helper falls back to local operation,
which keeps CI and CPU-only development workflows deterministic.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from omegaconf import OmegaConf

from src.algorithms.reward_model import GRPORewardModel, RewardBreakdown, ToolSpec


@dataclass(frozen=True)
class GRPOSample:
    """One rollout sample used for GRPO policy optimisation.

    中文说明：一个 GRPO 样本对应一次 rollout 的训练记录，包含原始 prompt、模型 response、
    生成时的 logprobs、基于 reward 归一化得到的 advantage，以及可选的 group/reward/metadata。
    后续 loss_fn 会读取这些字段来计算策略更新目标。
    """

    prompt: str
    response: str
    logprobs: Any
    advantage: float
    group_id: Any | None = None
    reward: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DistributedGRPOBuffer:
    """Simple append-only buffer for distributed GRPO samples.

    In a production Ray deployment this class can be wrapped by a Ray actor. The
    local implementation is deliberately serialisable and exposes enough methods
    for workers/trainers to exchange ``(prompt, response, logprobs, advantages)``
    payloads without depending on a specific replay-buffer library.
    """

    def __init__(self) -> None:
        # 内部使用普通 list 保存样本；真实分布式训练时可把该类包装成 Ray Actor。
        self._samples: list[GRPOSample] = []

    def add(
        self,
        prompt: str,
        response: str,
        logprobs: Any,
        advantage: float,
        *,
        group_id: Any | None = None,
        reward: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GRPOSample:
        # 将输入字段规范化为 GRPOSample，确保 advantage/reward 为 float，metadata 为普通 dict。
        sample = GRPOSample(
            prompt=prompt,
            response=response,
            logprobs=logprobs,
            advantage=float(advantage),
            group_id=group_id,
            reward=None if reward is None else float(reward),
            metadata=dict(metadata or {}),
        )
        self._samples.append(sample)
        return sample

    def extend(self, samples: Iterable[GRPOSample | Mapping[str, Any]]) -> None:
        for sample in samples:
            if isinstance(sample, GRPOSample):
                self._samples.append(sample)
            else:
                self.add(**sample)

    def sample_batch(self, batch_size: int | None = None, *, clear: bool = False) -> list[GRPOSample]:
        if batch_size is None or batch_size >= len(self._samples):
            batch = list(self._samples)
        else:
            batch = list(self._samples[:batch_size])
        if clear:
            del self._samples[: len(batch)]
        return batch

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)


class GRPOWorker:
    """Rollout worker that scores vLLM generations with ``GRPORewardModel``.

    中文说明：Worker 的职责很单一：接收 rollout 阶段生成的 response，调用奖励模型返回
    RewardBreakdown。这样可以让“生成”“奖励”“训练”三个部分解耦。
    """

    def __init__(
        self,
        reward_model: GRPORewardModel | None = None,
        *,
        reward_config: Mapping[str, Any] | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        # 避免同时传入已构造好的 reward_model 和构造 reward_model 所需配置，防止配置来源冲突。
        if reward_model is not None and (reward_config is not None or config_path is not None):
            raise ValueError("Pass either reward_model or reward_config/config_path, not both")
        self.reward_model = reward_model or build_reward_model_from_config(
            config_path=config_path,
            reward_config=reward_config,
        )

    def score_response(
        self,
        prompt: str,
        response: str,
        *,
        context: Mapping[str, Any] | None = None,
        outcome_reward: float = 0.0,
    ) -> RewardBreakdown:
        """Score one generated response and return a detailed breakdown."""

        return self.reward_model.compute_reward(
            response,
            prompt=prompt,
            context=context,
            outcome_reward=outcome_reward,
        )

    def score_batch(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        *,
        contexts: Sequence[Mapping[str, Any] | None] | None = None,
        outcome_rewards: Sequence[float] | None = None,
    ) -> list[RewardBreakdown]:
        """Score a batch of vLLM outputs from the rollout phase."""

        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")
        contexts = contexts or [None] * len(responses)
        outcome_rewards = outcome_rewards or [0.0] * len(responses)
        if len(contexts) != len(responses) or len(outcome_rewards) != len(responses):
            raise ValueError("contexts/outcome_rewards must match responses length")
        return [
            self.score_response(prompt, response, context=context, outcome_reward=outcome_reward)
            for prompt, response, context, outcome_reward in zip(prompts, responses, contexts, outcome_rewards)
        ]


def compute_advantages_distributed(
    local_rewards: Sequence[float | RewardBreakdown],
    *,
    group_ids: Sequence[Any] | None = None,
    reward_model: GRPORewardModel | None = None,
    ray_module: Any | None = None,
    gather_fn: Callable[[Any], Any] | None = None,
) -> list[float]:
    """Compute globally normalised advantages across Ray workers.

    中文说明：GRPO 通常会对同一 prompt 的多条候选回答做 group-wise 标准化。
    分布式场景下，每个 worker 只看到本地 reward，因此这里先尝试收集全局 reward，
    用全局统计量计算 advantage，再切回当前 worker 对应的本地结果。

    ``local_rewards`` contains the rewards visible to the current worker. In a
    real Ray run, pass ``gather_fn`` that returns all worker payloads, or rely on a
    Ray actor/collective wrapper to provide ``ray.get``-compatible object refs.
    The returned list is aligned to the current worker's local rewards while the
    normalisation statistics are computed over the gathered global rewards.
    """

    reward_model = reward_model or GRPORewardModel()
    local_values = _reward_values(local_rewards)
    local_group_ids = list(group_ids) if group_ids is not None else [0] * len(local_values)
    if len(local_group_ids) != len(local_values):
        raise ValueError("group_ids must have the same length as local_rewards")

    # payload 是当前 worker 与其他 worker 交换的最小信息：reward 数值和分组 id。
    payload = {"rewards": local_values, "group_ids": local_group_ids}
    gathered = gather_fn(payload) if gather_fn is not None else _try_ray_gather(payload, ray_module=ray_module)
    payloads = _normalise_gathered_payloads(gathered, fallback=payload)

    all_rewards: list[float] = []
    all_group_ids: list[Any] = []
    local_payload_index = 0
    for idx, item in enumerate(payloads):
        rewards = [float(v) for v in item["rewards"]]
        gids = list(item.get("group_ids") or [0] * len(rewards))
        if len(gids) != len(rewards):
            raise ValueError("gathered group_ids must match gathered rewards")
        if item is payload or item == payload:
            local_payload_index = idx
        all_rewards.extend(rewards)
        all_group_ids.extend(gids)

    # 先基于全局 reward 计算 advantage，再根据本地 payload 在 all-gather 结果中的 offset 截取回来。
    all_advantages = reward_model.compute_group_advantages(all_rewards, group_ids=all_group_ids)
    offset = sum(len(payloads[i]["rewards"]) for i in range(local_payload_index))
    return all_advantages[offset : offset + len(local_values)]


class GRPOTrainer:
    """Minimal veRL-style trainer with diagnostics and gradient accumulation.

    中文说明：这是训练主控类：
    1. 调用 worker/reward_model 给 rollout 打分；
    2. 计算 group-normalized advantage；
    3. 将样本写入 buffer；
    4. 调用外部注入的 loss_fn 与 optimizer 执行训练更新。
    代码保持框架无关，方便接入 torch、veRL 或单元测试中的 mock 对象。
    """

    def __init__(
        self,
        policy: Any | None = None,
        optimizer: Any | None = None,
        *,
        reward_model: GRPORewardModel | None = None,
        config: Mapping[str, Any] | None = None,
        config_path: str | Path | None = None,
        loss_fn: Callable[[Any, Sequence[GRPOSample]], Any] | None = None,
        empty_cache_every: int | None = None,
    ) -> None:
        # 读取默认 YAML 配置，并叠加外部 overrides。
        self.config = load_grpo_config(config_path=config_path, overrides=config)
        self.reward_model = reward_model or build_reward_model_from_config(reward_config=self.config.get("reward_model", {}))
        self.worker = GRPOWorker(self.reward_model)
        self.buffer = DistributedGRPOBuffer()
        self.policy = policy
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        train_cfg = self.config.get("trainer", {})
        # 梯度累积步数：显存不足时可用多个 micro-batch 模拟更大的有效 batch。
        self.gradient_accumulation_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
        self.empty_cache_every = empty_cache_every if empty_cache_every is not None else train_cfg.get("empty_cache_every", 1)
        self.global_step = 0
        self.optimizer_steps = 0
        self.last_diagnostics: dict[str, float] = {}

    def ingest_rollouts(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        logprobs: Sequence[Any],
        *,
        group_ids: Sequence[Any] | None = None,
        contexts: Sequence[Mapping[str, Any] | None] | None = None,
        outcome_rewards: Sequence[float] | None = None,
        gather_fn: Callable[[Any], Any] | None = None,
    ) -> list[GRPOSample]:
        """Score rollouts, compute advantages and store samples in the buffer."""

        if not (len(prompts) == len(responses) == len(logprobs)):
            raise ValueError("prompts, responses and logprobs must have the same length")
        group_ids = list(group_ids) if group_ids is not None else list(prompts)
        # 1) 对 rollout response 逐条打分，得到可解释的 RewardBreakdown。
        rewards = self.worker.score_batch(
            prompts,
            responses,
            contexts=contexts,
            outcome_rewards=outcome_rewards,
        )
        # 2) 对 reward 做 group-wise 标准化，得到 GRPO 训练使用的 advantage。
        advantages = compute_advantages_distributed(
            rewards,
            group_ids=group_ids,
            reward_model=self.reward_model,
            gather_fn=gather_fn,
        )
        samples: list[GRPOSample] = []
        # 3) 将 prompt/response/logprob/advantage/reward 明细写入训练 buffer。
        for prompt, response, lp, gid, breakdown, advantage in zip(prompts, responses, logprobs, group_ids, rewards, advantages):
            samples.append(
                self.buffer.add(
                    prompt,
                    response,
                    lp,
                    advantage,
                    group_id=gid,
                    reward=breakdown.final_reward,
                    metadata={"reward_breakdown": breakdown},
                )
            )
        self.log_iteration_diagnostics(rewards)
        return samples

    def train_batch(self, batch: Sequence[GRPOSample] | None = None) -> dict[str, float]:
        """Run one logical batch with gradient accumulation.

        ``loss_fn`` should return either a torch scalar tensor (with
        ``backward``) or a numeric loss. Optimizer stepping happens every
        ``gradient_accumulation_steps`` micro-batches, allowing large effective
        batch sizes on limited VRAM.
        """

        batch = list(batch or self.buffer.sample_batch(clear=True))
        if not batch:
            return {"loss": 0.0, "optimizer_steps": float(self.optimizer_steps)}
        if self.loss_fn is None:
            # CI-friendly default: a deterministic surrogate objective.
            # 中文说明：测试环境可能没有真实 policy/optimizer，这里用确定性的替代 loss 保证流程可跑通。
            avg_loss = -sum(sample.advantage for sample in batch) / len(batch)
            self.global_step += 1
            self._maybe_empty_cache()
            return {"loss": float(avg_loss), "optimizer_steps": float(self.optimizer_steps)}

        total_loss = 0.0
        # 将 batch 切成多个 micro-batch，逐个 backward，并按累积步数触发 optimizer.step()。
        micro_batches = [batch[i : i + self.gradient_accumulation_steps] for i in range(0, len(batch), self.gradient_accumulation_steps)]
        if self.optimizer is not None and hasattr(self.optimizer, "zero_grad"):
            self.optimizer.zero_grad()

        for micro_index, micro_batch in enumerate(micro_batches, start=1):
            loss = self.loss_fn(self.policy, micro_batch)
            scaled_loss = _scale_loss(loss, 1.0 / self.gradient_accumulation_steps)
            if hasattr(scaled_loss, "backward"):
                scaled_loss.backward()
            total_loss += _loss_to_float(loss)
            if micro_index % self.gradient_accumulation_steps == 0 or micro_index == len(micro_batches):
                if self.optimizer is not None and hasattr(self.optimizer, "step"):
                    self.optimizer.step()
                    self.optimizer_steps += 1
                if self.optimizer is not None and hasattr(self.optimizer, "zero_grad"):
                    self.optimizer.zero_grad()
                self._maybe_empty_cache()

        self.global_step += 1
        return {"loss": total_loss / len(micro_batches), "optimizer_steps": float(self.optimizer_steps)}

    def log_iteration_diagnostics(self, rewards: Sequence[RewardBreakdown]) -> dict[str, float]:
        """Print reward saturation diagnostics for the current batch."""

        # 记录 reward 方差、平均长度、平均 reward，用于观察奖励饱和或输出过长等训练现象。
        final_rewards = [float(r.final_reward) for r in rewards]
        lengths = [float(r.length) for r in rewards]
        mean_reward = sum(final_rewards) / len(final_rewards) if final_rewards else 0.0
        reward_variance = (
            sum((reward - mean_reward) ** 2 for reward in final_rewards) / len(final_rewards)
            if final_rewards
            else 0.0
        )
        average_length = sum(lengths) / len(lengths) if lengths else 0.0
        self.last_diagnostics = {
            "reward_variance": reward_variance,
            "average_length": average_length,
            "mean_reward": mean_reward,
        }
        print(
            "[GRPO Diagnostics] "
            f"reward_variance={reward_variance:.6f}, "
            f"average_length={average_length:.2f}, "
            f"mean_reward={mean_reward:.6f}"
        )
        return dict(self.last_diagnostics)

    def _maybe_empty_cache(self) -> None:
        if not self.empty_cache_every:
            return
        if self.global_step % int(self.empty_cache_every) != 0:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # Diagnostics/cache hooks must never crash CPU-only training.
            return


def load_grpo_config(
    config_path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load ``configs/grpo_config.yaml`` and merge optional overrides."""

    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "grpo_config.yaml"
    path = Path(config_path)
    base = OmegaConf.load(path) if path.exists() else OmegaConf.create({})
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.create(dict(overrides)))
    return OmegaConf.to_container(base, resolve=True) or {}


def build_reward_model_from_config(
    *,
    config_path: str | Path | None = None,
    reward_config: Mapping[str, Any] | None = None,
) -> GRPORewardModel:
    config = dict(reward_config or load_grpo_config(config_path).get("reward_model", {}))
    tool_specs = _build_tool_specs(config.pop("tool_specs", {}))
    return GRPORewardModel(tool_specs=tool_specs, **config)


def _build_tool_specs(raw_specs: Mapping[str, Any]) -> dict[str, ToolSpec]:
    specs: dict[str, ToolSpec] = {}
    for name, spec in raw_specs.items():
        spec = dict(spec or {})
        specs[name] = ToolSpec(
            required_params=set(spec.get("required_params", []) or []),
            optional_params=set(spec.get("optional_params", []) or []),
        )
    return specs


def _reward_values(rewards: Sequence[float | RewardBreakdown]) -> list[float]:
    return [float(r.final_reward if isinstance(r, RewardBreakdown) else r) for r in rewards]


def _try_ray_gather(payload: Mapping[str, Any], *, ray_module: Any | None = None) -> Any:
    if ray_module is None:
        try:
            ray_module = importlib.import_module("ray")
        except Exception:
            return [payload]
    # Without a project-specific Ray actor/collective group there is no safe
    # implicit all-gather API. If callers pass object refs through a fake/custom
    # module for tests, honour ``get``; otherwise return the local payload.
    try:
        if hasattr(ray_module, "get") and hasattr(ray_module, "_grpo_payload_refs"):
            return ray_module.get(ray_module._grpo_payload_refs + [payload])
    except Exception:
        pass
    return [payload]


def _normalise_gathered_payloads(gathered: Any, *, fallback: Mapping[str, Any]) -> list[MutableMapping[str, Any]]:
    if gathered is None:
        return [dict(fallback)]
    if isinstance(gathered, Mapping):
        gathered = [gathered]
    payloads: list[MutableMapping[str, Any]] = []
    for item in gathered:
        if isinstance(item, Mapping) and "rewards" in item:
            payloads.append(dict(item))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            payloads.append({"rewards": list(item), "group_ids": [0] * len(item)})
        else:
            raise TypeError("gather_fn must return payload dicts or reward sequences")
    return payloads or [dict(fallback)]


def _scale_loss(loss: Any, scale: float) -> Any:
    try:
        return loss * scale
    except TypeError:
        return float(loss) * scale


def _loss_to_float(loss: Any) -> float:
    if hasattr(loss, "detach"):
        loss = loss.detach()
    if hasattr(loss, "cpu"):
        loss = loss.cpu()
    if hasattr(loss, "item"):
        return float(loss.item())
    return float(loss)
