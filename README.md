# Agentic-GRPO-LongHorizon

面向长程工具调用智能体的工业级强化学习项目。项目以 **veRL 风格训练栈**、**GRPO（Group Relative Policy Optimization）**、**异步 vLLM Rollout** 和 **tau-bench Airline** 长程任务为核心，目标是在复杂多轮决策、状态追踪与 API 工具调用场景中提升 LLM Agent 的 `pass^1` 成功率。

> 当前仓库已完成从环境适配、奖励建模、token mask、异步采样到最终训练入口的端到端最小闭环；核心模块均配套单元测试，可在未安装完整 GPU 训练依赖的 CPU/CI 环境中验证主要逻辑。

## 项目定位

传统单轮问答评测难以反映真实 Agent 在业务系统中的能力。本项目关注更接近生产环境的长程任务：模型需要理解用户目标、规划多步行动、调用外部工具、根据环境反馈修正策略，并最终完成任务。

本项目构建一套可复现、可扩展的长程 Agent RL 实验框架，重点包括：

- **tau-bench Airline 环境适配**：将 tau-bench 航空客服任务封装为 veRL sampler 兼容的 rollout transition。
- **PRM-Lite 过程奖励**：对中间推理、工具调用合法性、观察证据一致性与重复推理进行稠密奖励/惩罚。
- **LATA 长度归一化**：通过 `reward / sqrt(length)` 抑制冗长无效推理，鼓励高效决策。
- **GRPO 组内优势归一化**：对同一 prompt 的多条采样结果进行组内标准化，提高训练稳定性。
- **Qwen2.5 Render-Twice-Diff Mask**：精确构造 response/thought/action 级 token loss mask，适配 chat template 漂移。
- **异步 vLLM Rollout**：支持并发采样、ContextVar 状态隔离、TPS 监控，并直接对接 trainer ingest 流程。
- **最终训练入口**：`scripts/train_agentic_grpo.py` 负责配置加载、Qwen2.5/vLLM 初始化、训练循环、checkpoint 和 tau-bench unseen 评估 hook。

## 当前已实现能力

### 1. tau-bench Airline 环境适配层

文件：[`envs/tau_env_wrapper.py`](envs/tau_env_wrapper.py)

`TauAirlineEnvWrapper` 将 tau-bench `AirlineEnv` 适配为 veRL sampler 常用 transition 格式：

- 输出 `observation` / `reward` / `done` / `info`。
- 兼容 tau-bench 多种 `reset()` / `step()` 返回格式。
- 使用 `contextvars.ContextVar` 隔离并发 rollout 的环境状态。
- 记录 episode id、step count、累计奖励、耗时、process metrics 等信息。
- 支持 `max_steps` 截断。

### 2. GRPO 奖励模型

文件：[`src/algorithms/reward_model.py`](src/algorithms/reward_model.py)

已实现：

- `GRPORewardModel`
- `ToolSpec`
- `RewardBreakdown`
- `StepReward`
- `ThoughtStep`
- `compute_reward()`
- `apply_lata()`
- `compute_group_advantages()`

核心逻辑包括：

- 解析模型输出中的 `Thought` / `Action` / `Tool` / `Observation` / `Final` 等结构化步骤。
- 校验工具调用是否包含必需参数、是否存在未知参数、是否调用不支持的工具。
- 对有证据支撑的推理给予过程奖励，对无依据数字或事实声明施加惩罚。
- 对重复推理步骤进行惩罚。
- 将环境最终奖励与过程奖励合并后进行 LATA 长度归一化。
- 对同一 prompt group 内的多条采样结果计算 GRPO advantage。

### 3. veRL/Ray 风格 GRPO Trainer

文件：[`src/algorithms/grpo_trainer.py`](src/algorithms/grpo_trainer.py)

已实现轻量、可测试的 GRPO 训练组件：

- `GRPOWorker`：持有 `GRPORewardModel`，对 vLLM rollout 文本打分。
- `compute_advantages_distributed()`：支持 Ray/gather 风格的全局 reward 聚合与 advantage 归一化；无 Ray 时自动本地回退。
- `DistributedGRPOBuffer`：存储 `prompt`、`response`、`logprobs`、`advantage`、`reward` 和 metadata。
- `GRPOTrainer.ingest_rollouts()`：完成 rollout 打分、advantage 计算与 buffer 写入。
- `GRPOTrainer.train_batch()`：支持 micro batch、gradient accumulation、optimizer step 和空缓存 hook。
- `log_iteration_diagnostics()`：输出 reward variance、average length 等训练诊断指标。
- `load_grpo_config()` / `build_reward_model_from_config()`：从 YAML/OmegaConf 配置构造奖励模型与训练参数。

### 4. Qwen2.5 Token Mask 工具

文件：[`src/utils/token_utils.py`](src/utils/token_utils.py)

实现 `get_grpo_masks()` 与 `GRPOMasks`：

- 使用 **Render-Twice-Diff** 策略：分别渲染 prompt-only conversation 与 prompt+assistant full conversation，定位 assistant response 起点。
- 生成与 `input_ids` 对齐的：
  - `attention_mask`
  - `response_mask`
  - `thought_mask`
  - `action_mask`
- 兼容 Qwen2.5 Instruct 风格 chat template。
- `offset_alignment()` 可修正少量 generation prompt/template token 漂移。

### 5. 异步 vLLM Rollout Adapter

文件：[`src/models/vllm_adapter.py`](src/models/vllm_adapter.py)

实现 `AsyncVLLMRollout` 与 `generate_and_score()`：

- 可懒加载真实 `vllm.AsyncLLMEngine`，也可在测试中注入 fake engine。
- 支持 batch 并发采样与 `max_concurrency` 限流。
- 使用 `ContextVar` 为每个异步请求隔离 `env_session_id` 与 tool context。
- 支持 vLLM coroutine / async iterator 风格输出。
- 标准化生成结果为 `VLLMGeneration` / `RolloutBatch`。
- 统计 total tokens、elapsed time、TPS。
- `generate_and_score()` 可直接调用 `GRPOTrainer.ingest_rollouts()`，把 rollout、奖励和样本写入训练 buffer。

### 6. 最终训练入口脚本

文件：[`scripts/train_agentic_grpo.py`](scripts/train_agentic_grpo.py)

`AgenticGRPOEntrypoint` 串联配置、rollout、trainer、checkpoint 和 unseen eval：

- 默认从 [`configs/grpo_config.yaml`](configs/grpo_config.yaml) 加载配置。
- 默认模型为 `Qwen/Qwen2.5-7B-Instruct`。
- 支持 OmegaConf dotlist override，例如 `trainer.total_steps=10`。
- 每步流程：
  1. 选择 tau-bench airline 风格 prompt batch。
  2. 调用 `generate_and_score()` 异步生成并打分。
  3. 调用 `GRPOTrainer.train_batch()` 执行一次逻辑训练更新。
  4. 打印 `overall pass^1`、`reward_variance`、`average_length`、`TPS`、`loss`。
- checkpoint hook 支持：
  - `policy.save_pretrained()`
  - `policy.save()`
  - 无 policy 时写入 `trainer_state.json`
- checkpoint 时触发 tau-bench unseen evaluation hook；无外部 evaluator 时使用 reward model 代理评估。

## 目录结构

```text
Agentic-GRPO-LongHorizon/
├── configs/
│   ├── .gitkeep
│   └── grpo_config.yaml              # GRPO reward/trainer 默认配置
├── docs/
│   └── .gitkeep
├── envs/
│   ├── __init__.py
│   └── tau_env_wrapper.py            # tau-bench AirlineEnv 的 veRL 适配层
├── scripts/
│   ├── __init__.py
│   └── train_agentic_grpo.py         # 最终训练入口脚本
├── src/
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── reward_model.py           # PRM-Lite / LATA / Group Advantage 奖励模块
│   │   └── grpo_trainer.py           # GRPO trainer、worker、buffer 与配置加载
│   ├── models/
│   │   ├── __init__.py
│   │   └── vllm_adapter.py           # Async vLLM rollout adapter
│   └── utils/
│       ├── __init__.py
│       └── token_utils.py            # Qwen2.5 Render-Twice-Diff token masks
├── tests/
│   ├── conftest.py
│   ├── test_reward_model.py
│   ├── test_tau_env_wrapper.py
│   ├── test_integration_tau.py
│   ├── test_grpo_trainer.py
│   ├── test_token_utils.py
│   ├── test_vllm_adapter.py
│   └── test_train_agentic_grpo.py
├── requirements.txt
└── README.md
```

## 环境要求

建议环境：

- Python 3.10+
- PyTorch 2.4+
- CUDA 环境（如需运行 vLLM / 大模型训练）
- Windows / Linux 均可进行单元测试；完整训练建议使用 Linux + GPU 环境

核心依赖见 [`requirements.txt`](requirements.txt)：

```text
verl
vllm
torch>=2.4
flash-attn
tau-bench
omegaconf
ray
```

> 注意：`flash-attn`、`vllm`、`verl` 等依赖对 CUDA、PyTorch 与系统环境版本较敏感。若只进行本仓库单元测试，可先安装 pytest 与 omegaconf，并根据需要逐步安装完整训练依赖。

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/zyjcode00/Agentic-GRPO-LongHorizon.git
cd Agentic-GRPO-LongHorizon
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
pip install pytest
```

如果遇到 GPU 相关依赖安装问题，建议先参考对应项目文档安装 PyTorch、vLLM、flash-attn 和 veRL。仅运行单元测试时，可在轻量环境中安装 `pytest omegaconf` 后运行大部分测试。

### 4. 运行测试

```bash
pytest tests
```

也可以按模块运行：

```bash
pytest tests/test_tau_env_wrapper.py
pytest tests/test_reward_model.py tests/test_integration_tau.py
pytest tests/test_grpo_trainer.py
pytest tests/test_token_utils.py
pytest tests/test_vllm_adapter.py
pytest tests/test_train_agentic_grpo.py
```

当前测试覆盖：

- tau-bench Airline 环境 wrapper 的 reset / step 接口规范。
- 并发 rollout 场景下的 ContextVar 状态隔离。
- PRM-Lite 对合法/非法工具调用的奖励差异。
- LATA 对长输出的归一化惩罚。
- GRPO group advantage 的组内标准化与零方差处理。
- veRL/Ray 风格 trainer 的 reward scoring、distributed advantage、buffer、diagnostics 与 gradient accumulation。
- Qwen2.5 Render-Twice-Diff response/thought/action mask 对齐。
- Async vLLM rollout 并发采样、异步迭代、TPS 监控与 `GRPOTrainer.ingest_rollouts()` 对接。
- 最终入口脚本的配置加载、训练循环、checkpoint 与 tau-bench unseen eval hook。

## 配置说明

默认配置文件：[`configs/grpo_config.yaml`](configs/grpo_config.yaml)

```yaml
reward_model:
  correct_tool_reward: 0.4
  invalid_tool_penalty: -0.6
  reasoning_reward: 0.15
  unsupported_reasoning_penalty: -0.1
  observation_reward: 0.05
  repetition_penalty: -0.2
  max_step_reward: 1.0
  length_normalization: true
  min_length: 1
  std_epsilon: 1.0e-8
  tool_specs:
    search_flights:
      required_params: [origin, destination]
      optional_params: [date, cabin]
    book_reservation:
      required_params: [flight_id, passenger_name]
      optional_params: [seat]

trainer:
  gradient_accumulation_steps: 2
  empty_cache_every: 1
  train_batch_size: 8
  micro_batch_size: 2
```

可通过 CLI override 覆盖配置：

```bash
python scripts/train_agentic_grpo.py \
  --config configs/grpo_config.yaml \
  --override trainer.total_steps=10 \
  --override rollout.batch_size=4 \
  --override checkpoint.save_every=5
```

## 使用示例

### 环境适配层

```python
from envs.tau_env_wrapper import TauAirlineEnvWrapper

wrapper = TauAirlineEnvWrapper(max_steps=30)
transition = wrapper.reset(task_id="airline-task-001")

while not transition["done"]:
    action = {"tool": "search_flights", "arguments": {"origin": "SFO", "destination": "JFK"}}
    transition = wrapper.step(action)
```

### 奖励计算

```python
from src.algorithms.reward_model import GRPORewardModel, ToolSpec

reward_model = GRPORewardModel(
    tool_specs={
        "search_flights": ToolSpec(
            required_params={"origin", "destination"},
            optional_params={"date"},
        )
    }
)

result = reward_model.compute_reward(
    output="""
Thought: 用户需要从 SFO 到 JFK 的航班。
Action: search_flights(origin='SFO', destination='JFK')
Observation: 找到 3 个可选航班。
Final: 已为用户提供可选航班。
""",
    prompt="用户想查询 SFO 到 JFK 的航班。",
    outcome_reward=1.0,
)

print(result.raw_reward, result.final_reward)
```

### 组内 Advantage

```python
from src.algorithms.reward_model import compute_group_advantages

advantages = compute_group_advantages(
    rewards=[1.0, 2.0, 3.0, 10.0, 20.0],
    group_ids=["prompt-a", "prompt-a", "prompt-a", "prompt-b", "prompt-b"],
)
```

### Token Mask 构造

```python
from transformers import AutoTokenizer
from src.utils.token_utils import get_grpo_masks

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

masks = get_grpo_masks(
    tokenizer,
    prompt_messages=[{"role": "user", "content": "Search a flight from SFO to JFK."}],
    response="<thought>Need to search first.</thought>\nAction: search_flights(origin='SFO', destination='JFK')",
)

batch_item = masks.to_dict()
```

### 异步 Rollout + Trainer

```python
import asyncio
from src.algorithms.grpo_trainer import GRPOTrainer
from src.models.vllm_adapter import AsyncVLLMRollout, generate_and_score

async def main():
    rollout = AsyncVLLMRollout(
        model="Qwen/Qwen2.5-7B-Instruct",
        sampling_params={"temperature": 1.0, "max_tokens": 256},
        max_concurrency=8,
    )
    trainer = GRPOTrainer(config_path="configs/grpo_config.yaml")
    batch = await generate_and_score(
        rollout,
        ["You are a tau-bench airline assistant. Search flights from SFO to JFK."],
        trainer=trainer,
    )
    metrics = trainer.train_batch(batch.samples)
    print(batch.tps, metrics)

asyncio.run(main())
```

### 启动训练入口

```bash
python scripts/train_agentic_grpo.py \
  --config configs/grpo_config.yaml \
  --steps 10 \
  --checkpoint-every 5
```

运行时会打印类似诊断信息：

```text
[Train Diagnostics] step=1, overall pass^1=0.5000, reward_variance=0.012345, average_length=42.00, TPS=128.50, loss=0.000000
[Checkpoint] step=5, path=checkpoints/agentic_grpo/step_000005, tau_bench_unseen_success_rate=0.5000
```

## 研发路线图

- [x] 初始化项目目录与依赖声明。
- [x] 实现 tau-bench Airline 环境 veRL 适配层。
- [x] 实现 PRM-Lite / LATA / Group Advantage 奖励模块。
- [x] 添加 tau-bench 风格集成测试。
- [x] 实现 veRL/Ray 风格 `GRPOTrainer`、worker、buffer 与诊断指标。
- [x] 实现 Qwen2.5 Render-Twice-Diff token mask。
- [x] 实现 Async vLLM rollout adapter、ContextVar 隔离与 TPS 监控。
- [x] 实现最终训练入口脚本、checkpoint 与 tau-bench unseen eval hook。
- [ ] 接入真实 veRL 分布式 worker / Ray actor 集群。
- [ ] 接入真实 tau-bench Airline 官方评测流程与 unseen split。
- [ ] 输出 baseline、ablation 与训练曲线。

## 设计原则

1. **长程任务优先**：面向真实业务中的多步工具调用，而非只优化短问答。
2. **过程监督优先**：通过 PRM-Lite 提供更密集的中间反馈，降低稀疏终局奖励训练难度。
3. **工程可复现**：所有核心模块配套测试，训练配置与实验流程尽可能标准化。
4. **依赖可降级**：核心逻辑在不安装 vLLM、veRL、Ray、CUDA 的情况下仍可导入和测试。
5. **可扩展环境接口**：先支持 tau-bench Airline，后续可扩展至 retail、web task、database task 等更多工具环境。

## License

本项目当前以研究与工程实验为目标。具体开源协议请以仓库中的 LICENSE 文件为准。
