"""Agentic GRPO 训练入口脚本：用于在 tau-bench 风格任务上训练工具调用型 Agent。

Final entrypoint for Agentic GRPO training on tau-bench style tasks.

本文件可以理解为整个项目的“总调度器”：
1. 读取配置文件与命令行覆盖参数；
2. 构建 vLLM 异步 rollout 生成器；
3. 构建 GRPOTrainer 训练器；
4. 循环执行“生成回答/轨迹 -> 奖励打分 -> GRPO 更新”；
5. 按间隔保存 checkpoint，并在 unseen 任务上做简单评估。

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

中文说明：
这里的 “duck-typed” 指不强依赖某个具体类，只要对象“长得像”、拥有对应方法就调用。
例如保存模型时，如果 policy 有 save_pretrained 方法就用它；否则看是否有 save 方法；
再不行就退化为保存一份 JSON 元数据。这样能让脚本兼容不同训练框架或 mock 测试环境。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf

# 项目根目录：当前文件位于 scripts/train_agentic_grpo.py，parents[1] 即项目根目录。
# 例如：D:\LLM\Agentic-GRPO-LongHorizon
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 将项目根目录加入 Python 模块搜索路径，确保可以直接 import src.xxx。
# 这样无论从哪个工作目录启动脚本，都能找到项目内部模块。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# GRPOTrainer：负责把 rollout 样本放入 buffer、计算 advantage、执行训练更新。
# load_grpo_config：读取项目的 GRPO 配置文件。
from src.algorithms.grpo_trainer import GRPOTrainer, load_grpo_config

# AsyncVLLMRollout：基于 vLLM 的异步生成器，负责批量生成模型回答。
# RolloutBatch：一次 rollout 的批量结果数据结构。
# generate_and_score：封装“生成回答 + 调用 trainer/reward model 打分 + 写入训练 buffer”的流程。
from src.models.vllm_adapter import AsyncVLLMRollout, RolloutBatch, generate_and_score

# 如果配置文件里没有指定模型，默认使用 Qwen2.5-7B-Instruct。
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 当配置中没有提供训练 prompts 时，使用下面两个航空客服任务作为 smoke test。
# smoke test 的意义是：即使没有真实数据，也能快速检查训练主流程能否跑通。
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

# unseen prompts 用于 checkpoint 后的简单泛化评估。
# “unseen” 表示这些任务不是训练 prompts，用来粗略观察模型是否只记住训练样本。
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
    """围绕 rollout 生成与 GRPO 更新的轻量编排层。

    这个类不是具体算法本身，而是把项目中的几个关键组件串起来：
    - rollout：调用模型批量生成回答/轨迹；
    - trainer：对回答打分、缓存样本、执行 GRPO 训练；
    - evaluator：可选的外部评估器，用于 tau-bench unseen 评估。
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        rollout: AsyncVLLMRollout | None = None,
        trainer: GRPOTrainer | None = None,
        evaluator: Any | None = None,
    ) -> None:
        # 将传入配置转为普通 dict，避免外部 Mapping 类型差异影响后续处理。
        self.config = dict(config)

        # 如果外部没有注入 rollout，就根据配置创建一个 AsyncVLLMRollout。
        # 支持注入的好处：测试时可以传入 mock rollout，避免真的启动大模型。
        self.rollout = rollout or build_rollout(self.config)

        # 如果外部没有注入 trainer，就创建项目默认的 GRPOTrainer。
        self.trainer = trainer or GRPOTrainer(config=self.config)

        # evaluator 是可选组件；如果没有，后面会使用 reward model 做一个代理评估。
        self.evaluator = evaluator

    async def train_loop(self) -> list[dict[str, float]]:
        """执行完整训练循环，并返回每一步的指标历史。

        每个 step 的核心流程是：
        1. 选择当前 batch 的 prompts；
        2. 构造上下文、group id、环境 session id；
        3. 调用 generate_and_score 生成回答并计算奖励；
        4. 调用 trainer.train_batch() 做一次训练更新；
        5. 汇总并打印诊断指标；
        6. 如到达 checkpoint 间隔，则保存模型并评估 unseen 成功率。
        """

        # 从总配置中拆出几个子配置，若不存在则使用空 dict。
        train_cfg = dict(self.config.get("trainer", {}) or {})
        rollout_cfg = dict(self.config.get("rollout", {}) or {})
        checkpoint_cfg = dict(self.config.get("checkpoint", {}) or {})

        # 训练总步数：优先 trainer.total_steps，其次兼容 trainer.max_steps，默认 1。
        total_steps = int(train_cfg.get("total_steps", train_cfg.get("max_steps", 1)))

        # checkpoint 保存间隔：优先 checkpoint.save_every，其次 trainer.checkpoint_every。
        # 如果结果为 0，则表示不在训练中自动保存 checkpoint。
        checkpoint_every = int(checkpoint_cfg.get("save_every", train_cfg.get("checkpoint_every", 0)) or 0)

        # checkpoint 输出目录：配置优先，否则保存到 checkpoints/agentic_grpo。
        output_dir = Path(checkpoint_cfg.get("output_dir", train_cfg.get("output_dir", "checkpoints/agentic_grpo")))

        # 加载训练 prompts：可能来自配置内联列表、数据文件，或默认 fallback prompts。
        prompts = load_prompts(self.config, split="train")

        # rollout batch size：优先 rollout.batch_size，其次 trainer.train_batch_size，默认等于 prompts 数量。
        batch_size = int(rollout_cfg.get("batch_size", train_cfg.get("train_batch_size", len(prompts))) or len(prompts))

        # 保存每个训练 step 的指标，方便调用方做测试或进一步分析。
        metrics_history: list[dict[str, float]] = []

        # 主训练循环：step 从 1 开始，便于日志和 checkpoint 命名。
        for step in range(1, total_steps + 1):
            # 按 step 和 batch_size 从 prompts 中循环取样。
            # 如果 prompts 数少于训练步数，会通过取模重复使用。
            batch_prompts = select_prompt_batch(prompts, step=step, batch_size=batch_size)

            # group_ids 用于 GRPO 的“组”概念。
            # GRPO 通常会比较同一个 prompt 或同一组样本内部的奖励，计算相对 advantage。
            group_ids = [f"train-{step}-{i}" for i in range(len(batch_prompts))]

            # contexts 给 reward model / 工具环境提供额外信息，如 split、step、工具上下文等。
            contexts = [build_context(prompt, split="train", step=step, index=i) for i, prompt in enumerate(batch_prompts)]

            # env_session_ids 标识每个样本对应的环境会话。
            # 对工具调用型 Agent 来说，不同样本应使用不同环境 session，避免状态串扰。
            env_session_ids = [f"train-step-{step}-env-{i}" for i in range(len(batch_prompts))]

            # 生成并打分：
            # - self.rollout 负责调用模型生成；
            # - self.trainer / trainer.worker 负责奖励打分；
            # - 生成出的样本通常会被写入 trainer 的经验 buffer，供后续 train_batch 使用。
            rollout_batch = await generate_and_score(
                self.rollout,
                batch_prompts,
                trainer=self.trainer,
                sampling_params=rollout_cfg.get("sampling_params"),
                contexts=contexts,
                group_ids=group_ids,
                env_session_ids=env_session_ids,
            )

            # 执行一次 GRPO 训练更新。
            # 具体逻辑在 src/algorithms/grpo_trainer.py 中，包括 advantage、loss、optimizer step 等。
            train_metrics = self.trainer.train_batch()

            # 将 rollout 结果、trainer 诊断信息、训练 loss 等汇总为统一指标。
            step_metrics = summarize_iteration(
                step=step,
                rollout_batch=rollout_batch,
                trainer=self.trainer,
                train_metrics=train_metrics,
            )
            metrics_history.append(step_metrics)

            # 打印当前 step 的关键信息，方便在命令行观察训练状态。
            print_diagnostics(step_metrics)

            # 如果设置了 checkpoint_every，并且当前 step 到达保存间隔，则保存 checkpoint 并做 unseen 评估。
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

        # 返回完整指标历史，便于单元测试或上层脚本收集结果。
        return metrics_history


def build_rollout(config: Mapping[str, Any]) -> AsyncVLLMRollout:
    """根据配置构建 AsyncVLLMRollout。

    该函数主要处理三类配置：
    - model：模型名称；
    - vllm.engine_args / rollout.engine_args：vLLM 引擎参数；
    - vllm.sampling_params / rollout.sampling_params：采样参数。
    """

    model_cfg = dict(config.get("model", {}) or {})
    rollout_cfg = dict(config.get("rollout", {}) or {})
    vllm_cfg = dict(config.get("vllm", {}) or {})

    # 按优先级寻找模型名：
    # 1. model.name
    # 2. model.model_name
    # 3. rollout.model
    # 4. vllm.model
    # 5. DEFAULT_MODEL
    model_name = (
        model_cfg.get("name")
        or model_cfg.get("model_name")
        or rollout_cfg.get("model")
        or vllm_cfg.get("model")
        or DEFAULT_MODEL
    )

    # engine_args 会传给 vLLM 引擎，比如 tensor_parallel_size、gpu_memory_utilization 等。
    # 这里优先使用 vllm.engine_args，其次 rollout.engine_args。
    engine_args = dict(vllm_cfg.get("engine_args", {}) or rollout_cfg.get("engine_args", {}) or {})

    # 确保 engine_args 中一定包含 model 字段。
    # 如果配置已提供 model，则不覆盖；否则填入上面解析出的 model_name。
    engine_args.setdefault("model", model_name)

    # 采样参数控制生成行为，如 temperature、max_tokens、top_p 等。
    sampling_params = dict(vllm_cfg.get("sampling_params", {}) or rollout_cfg.get("sampling_params", {}) or {})

    # 如果配置完全没有采样参数，则给一个保守默认值，避免 vLLM 调用缺少必要参数。
    if not sampling_params:
        sampling_params = {"temperature": 1.0, "max_tokens": 256}

    # 创建异步 rollout 组件。
    return AsyncVLLMRollout(
        model=model_name,
        engine_args=engine_args,
        sampling_params=sampling_params,
        max_concurrency=rollout_cfg.get("max_concurrency", vllm_cfg.get("max_concurrency")),
        print_tps=bool(rollout_cfg.get("print_tps", True)),
    )


def load_config(config_path: str | Path, overrides: Sequence[str] | None = None) -> dict[str, Any]:
    """加载 GRPO 配置，并应用命令行传入的 OmegaConf dotlist 覆盖项。

    示例：
    python scripts/train_agentic_grpo.py --override trainer.total_steps=10

    上面的 override 会把配置里的 trainer.total_steps 改为 10。
    """

    # 先读取 YAML 配置文件。
    config = load_grpo_config(config_path=config_path)

    # 如果传入了覆盖项，则使用 OmegaConf 的 dotlist 机制合并。
    if overrides:
        override_conf = OmegaConf.from_dotlist(list(overrides))
        merged = OmegaConf.merge(OmegaConf.create(config), override_conf)
        config = OmegaConf.to_container(merged, resolve=True) or {}

    # 返回普通 dict，降低后续代码对 OmegaConf 类型的依赖。
    return dict(config)


def load_prompts(config: Mapping[str, Any], *, split: str) -> list[str]:
    """根据 split 加载训练或评估 prompts。

    split="train" 时，优先读取：
    1. rollout.prompts
    2. data.train_prompts
    3. data.prompts
    4. data.train_path / rollout.prompt_path 指向的文件
    5. DEFAULT_TAU_AIRLINE_PROMPTS

    split="unseen" 时，优先读取：
    1. eval.unseen_prompts
    2. data.unseen_prompts
    3. eval.unseen_path / data.unseen_path 指向的文件
    4. DEFAULT_UNSEEN_PROMPTS
    """

    data_cfg = dict(config.get("data", {}) or {})
    rollout_cfg = dict(config.get("rollout", {}) or {})
    eval_cfg = dict(config.get("eval", {}) or {})

    # 根据 train / unseen 选择不同配置字段和 fallback 数据。
    if split == "train":
        inline = rollout_cfg.get("prompts") or data_cfg.get("train_prompts") or data_cfg.get("prompts")
        path = data_cfg.get("train_path") or rollout_cfg.get("prompt_path")
        fallback = DEFAULT_TAU_AIRLINE_PROMPTS
    else:
        inline = eval_cfg.get("unseen_prompts") or data_cfg.get("unseen_prompts")
        path = eval_cfg.get("unseen_path") or data_cfg.get("unseen_path")
        fallback = DEFAULT_UNSEEN_PROMPTS

    # 如果配置中直接给了 prompts 列表，则直接转成字符串列表返回。
    if inline:
        return [str(item) for item in inline]

    # 如果配置中给了文件路径，则从文件读取 prompts。
    if path:
        return read_prompt_file(Path(path))

    # 如果都没有，则使用默认 fallback prompts。
    return list(fallback)


def read_prompt_file(path: Path) -> list[str]:
    """从文本文件或 JSONL 文件读取 prompts。

    支持两种格式：
    - .jsonl：每行一个 JSON，优先读取 prompt 字段，其次 instruction 字段；
    - 其他后缀：每个非空行作为一个 prompt。
    """

    # 相对路径默认相对于项目根目录解析，避免受当前 shell 工作目录影响。
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    # JSONL 格式：逐行解析 JSON。
    if path.suffix.lower() == ".jsonl":
        prompts = []
        for line in path.read_text(encoding="utf-8").splitlines():
            # 跳过空行。
            if not line.strip():
                continue
            item = json.loads(line)

            # 兼容常见字段名：prompt / instruction。
            # 如果两者都没有，就把整个 JSON 对象转成字符串。
            prompts.append(str(item.get("prompt") or item.get("instruction") or item))
        return prompts

    # 普通文本格式：每个非空行都是一个 prompt。
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_prompt_batch(prompts: Sequence[str], *, step: int, batch_size: int) -> list[str]:
    """根据 step 从 prompts 中循环选择一个 batch。

    例如 prompts 有 3 条、batch_size=2：
    - step=1 取第 0、1 条；
    - step=2 取第 2、0 条；
    - step=3 取第 1、2 条。

    这种循环取样可以保证即使 prompts 很少，训练循环也不会越界。
    """

    if not prompts:
        raise ValueError("No prompts available for training")

    # batch_size 至少为 1，防止配置成 0 或负数导致空 batch。
    batch_size = max(1, int(batch_size))

    # 计算当前 step 的起始位置。step 从 1 开始，所以先减 1。
    start = ((step - 1) * batch_size) % len(prompts)

    # 通过取模循环访问 prompts，保证 offset 超过长度时能回到开头。
    return [str(prompts[(start + offset) % len(prompts)]) for offset in range(batch_size)]


def build_context(prompt: str, *, split: str, step: int, index: int) -> dict[str, Any]:
    """为单条 prompt 构造上下文信息。

    context 会被传给 rollout / reward model / 工具环境。
    对工具调用任务来说，奖励模型往往需要知道当前属于 train 还是 unseen、
    可用工具有哪些、benchmark 类型是什么等信息。
    """

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
    """汇总单个训练 step 的诊断指标。

    返回值统一为 dict[str, float]，包括：
    - step：当前步数；
    - overall_pass@1：奖励大于 0 的样本比例；
    - reward_variance：奖励方差；
    - average_length：平均生成长度；
    - TPS：tokens per second，吞吐量；
    - loss：训练损失；
    - optimizer_steps：优化器累计更新次数。
    """

    # trainer.last_diagnostics 通常由 GRPOTrainer 在处理 batch 时写入。
    diagnostics = dict(getattr(trainer, "last_diagnostics", {}) or {})

    # 取出 rollout 样本中非 None 的 reward。
    rewards = [sample.reward for sample in rollout_batch.samples if sample.reward is not None]

    # 简单定义 pass@1：reward > 0 视为成功。
    # 如果没有 reward，则成功率记为 0。
    overall_pass_at_1 = sum(1 for reward in rewards if float(reward) > 0.0) / len(rewards) if rewards else 0.0

    # 合并各类指标，并全部转成 float，方便日志、JSON 序列化和测试断言。
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
    """以统一格式打印训练诊断信息。"""

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
    """保存训练 checkpoint，并返回 checkpoint 目录路径。

    这里采用“能力探测”的保存策略：
    1. 如果 trainer.policy 有 save_pretrained 方法，按 HuggingFace 风格保存；
    2. 否则如果有 save 方法，调用 save；
    3. 如果都没有，就保存一份 trainer_state.json 元数据。

    这样做可以兼容真实模型、轻量 mock 对象和不同训练框架。
    """

    # 确保输出根目录存在。
    output_dir.mkdir(parents=True, exist_ok=True)

    # 每个 step 单独一个目录，例如 step_000100。
    checkpoint_dir = output_dir / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 从 trainer 上尝试获取 policy。
    policy = getattr(trainer, "policy", None)

    # HuggingFace Transformers 模型通常提供 save_pretrained。
    if policy is not None and hasattr(policy, "save_pretrained"):
        policy.save_pretrained(str(checkpoint_dir))

    # 某些自定义 policy 可能只提供 save。
    elif policy is not None and hasattr(policy, "save"):
        policy.save(str(checkpoint_dir))

    # 如果没有可保存的模型对象，就至少保存训练状态元数据，保证 checkpoint hook 有产物。
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
    """在 unseen prompts 上评估当前模型的成功率。

    评估分两种情况：
    1. 如果外部传入 evaluator，则优先调用 evaluator.evaluate_unseen；
    2. 如果没有 evaluator，则用当前 rollout 生成回答，再用 trainer.worker.score_response 打分，
       将 final_reward 大于 success_threshold 的样本视为成功。

    注意：第二种方式是一个代理评估，不等同于完整 tau-bench 官方评估，
    但能在没有完整环境时提供一个稳定可运行的指标。
    """

    eval_cfg = dict(config.get("eval", {}) or {})

    # 加载 unseen prompts，来源可能是配置、文件或默认 fallback。
    prompts = load_prompts(config, split="unseen")

    # 限制评估样本数，避免每次 checkpoint 评估过慢。
    max_cases = int(eval_cfg.get("max_unseen_cases", len(prompts)) or len(prompts))
    prompts = prompts[:max_cases]

    # 如果用户提供了外部 evaluator，则使用它进行评估。
    if evaluator is not None:
        result = evaluator.evaluate_unseen(prompts=prompts, rollout=rollout, trainer=trainer)

        # evaluator 可能是异步实现，也可能是同步实现；这里同时兼容。
        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            result = await result

        # evaluator 返回 dict 时，兼容 success_rate / unseen_success_rate 两种字段名。
        if isinstance(result, Mapping):
            return float(result.get("success_rate", result.get("unseen_success_rate", 0.0)))

        # evaluator 也可以直接返回一个数值。
        return float(result)

    # 没有 evaluator 时，使用 rollout + reward model 做代理评估。
    contexts = [build_context(prompt, split="unseen", step=0, index=i) for i, prompt in enumerate(prompts)]
    env_session_ids = [f"unseen-eval-env-{i}" for i in range(len(prompts))]

    # 生成 unseen prompts 的回答。
    generations = await rollout.generate(
        prompts,
        sampling_params=eval_cfg.get("sampling_params"),
        contexts=contexts,
        group_ids=[f"unseen-{i}" for i in range(len(prompts))],
        env_session_ids=env_session_ids,
    )

    # 逐条用 reward model 打分，并统计成功数量。
    successes = 0
    for generation in generations:
        reward = trainer.worker.score_response(generation.prompt, generation.response, context=generation.context)
        if reward.final_reward > float(eval_cfg.get("success_threshold", 0.0)):
            successes += 1

    # 返回成功率；如果没有生成结果，则返回 0。
    return successes / len(generations) if generations else 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Train Agentic GRPO with Qwen2.5 + async vLLM rollouts")

    # 配置文件路径，默认使用项目 configs/grpo_config.yaml。
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "grpo_config.yaml"))

    # 可重复传入的 OmegaConf dotlist 覆盖项，例如：
    # --override trainer.total_steps=10 --override rollout.batch_size=4
    parser.add_argument("--override", action="append", default=[], help="OmegaConf dotlist override, e.g. trainer.total_steps=10")

    # 便捷参数：直接覆盖 trainer.total_steps。
    parser.add_argument("--steps", type=int, default=None, help="Override trainer.total_steps")

    # 便捷参数：直接覆盖 checkpoint.save_every。
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Override checkpoint.save_every")

    return parser.parse_args(argv)


async def async_main(argv: Sequence[str] | None = None) -> list[dict[str, float]]:
    """异步主函数：解析参数、加载配置、创建入口对象并启动训练。"""

    args = parse_args(argv)

    # 收集命令行覆盖项。
    overrides = list(args.override or [])

    # 将 --steps 转换成 OmegaConf dotlist 覆盖 trainer.total_steps。
    if args.steps is not None:
        overrides.append(f"trainer.total_steps={args.steps}")

    # 将 --checkpoint-every 转换成 OmegaConf dotlist 覆盖 checkpoint.save_every。
    if args.checkpoint_every is not None:
        overrides.append(f"checkpoint.save_every={args.checkpoint_every}")

    # 加载配置，并应用所有覆盖项。
    config = load_config(args.config, overrides=overrides)

    # 构建入口编排对象，内部会创建 rollout 和 trainer。
    entrypoint = AgenticGRPOEntrypoint(config)

    # 启动训练循环。
    return await entrypoint.train_loop()


def main(argv: Sequence[str] | None = None) -> list[dict[str, float]]:
    """同步入口：用 asyncio.run 启动异步训练主函数。"""

    return asyncio.run(async_main(argv))


# 当文件被直接执行时进入训练；被 pytest/import 时不会自动运行。
if __name__ == "__main__":  # pragma: no cover
    main()
