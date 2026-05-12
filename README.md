# Agentic-GRPO-LongHorizon

面向长程工具调用智能体的工业级强化学习项目。项目以 **veRL** 训练栈、**GRPO（Group Relative Policy Optimization）** 算法和 **tau-bench Airline** 长程任务为核心，目标是在复杂多轮决策、状态追踪与 API 工具调用场景中提升 LLM Agent 的 `pass^1` 成功率。

> 当前仓库处于早期研发阶段：已完成项目骨架、tau-bench Airline 环境适配层，以及 GRPO 奖励建模模块的基础实现与单元测试。

## 项目定位

传统单轮问答评测难以反映真实 Agent 在业务系统中的能力。本项目关注更接近生产环境的长程任务：模型需要理解用户目标、规划多步行动、调用外部工具、根据环境反馈修正策略，并最终完成任务。

本项目计划构建一套可复现、可扩展的长程 Agent RL 实验框架，重点包括：

- **tau-bench Airline 环境适配**：将 tau-bench 航空客服任务封装为 veRL 兼容的 rollout 接口。
- **异步 Rollout 支持**：面向长轨迹和工具交互场景，支持并发采样时的环境状态隔离。
- **PRM-Lite 过程奖励**：不只依赖最终结果奖励，而是对中间推理、工具调用合法性和证据一致性提供稠密反馈。
- **LATA 长度归一化**：通过 `reward / sqrt(length)` 抑制冗长无效推理，鼓励高效决策。
- **GRPO 组内优势归一化**：对同一 prompt 的多条采样结果进行组内标准化，提升训练稳定性。

## 当前已实现能力

### 1. tau-bench Airline 环境适配层

文件：[`envs/tau_env_wrapper.py`](envs/tau_env_wrapper.py)

已实现 `TauAirlineEnvWrapper`：

- 兼容 veRL sampler 常用 transition 格式：
  - `observation`
  - `reward`
  - `done`
  - `info`
- 支持 tau-bench `AirlineEnv` 的多种 `reset()` / `step()` 返回格式。
- 通过 `contextvars.ContextVar` 隔离并发 rollout 的环境状态，避免多个异步任务共享同一 wrapper 时互相污染。
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

- 解析模型输出中的 Thought / Action / Tool / Observation / Final 等结构化步骤。
- 校验工具调用是否包含必需参数、是否存在未知参数、是否调用不支持的工具。
- 对有证据支撑的推理给予过程奖励，对无依据数字或事实声明施加惩罚。
- 对重复推理步骤进行惩罚。
- 将环境最终奖励与过程奖励合并后进行 LATA 长度归一化。
- 对同一 prompt group 内的多条采样结果计算 GRPO advantage。

## 目录结构

```text
Agentic-GRPO-LongHorizon/
├── configs/                    # Hydra/OmegaConf 配置，后续放置训练与实验配置
├── docs/                       # 设计文档与技术说明
├── envs/
│   ├── __init__.py
│   └── tau_env_wrapper.py      # tau-bench AirlineEnv 的 veRL 适配层
├── scripts/                    # 训练、评估、rollout 与工具脚本
├── src/
│   ├── __init__.py
│   └── algorithms/
│       ├── __init__.py
│       └── reward_model.py     # PRM-Lite / LATA / Group Advantage 奖励模块
├── tests/
│   ├── conftest.py
│   ├── test_reward_model.py
│   └── test_tau_env_wrapper.py
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

> 注意：`flash-attn`、`vllm`、`verl` 等依赖对 CUDA、PyTorch 与系统环境版本较敏感。若只进行本仓库单元测试，可先安装 pytest，并根据需要逐步安装完整训练依赖。

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

如果遇到 GPU 相关依赖安装问题，建议先参考对应项目文档安装 PyTorch、vLLM、flash-attn 和 veRL。

### 4. 运行测试

```bash
pytest tests
```

当前测试覆盖：

- tau-bench Airline 环境 wrapper 的 reset / step 接口规范。
- 并发 rollout 场景下的 contextvars 状态隔离。
- PRM-Lite 对合法/非法工具调用的奖励差异。
- LATA 对长输出的归一化惩罚。
- GRPO group advantage 的组内标准化与零方差处理。

## 使用示例

### 环境适配层

```python
from envs.tau_env_wrapper import TauAirlineEnvWrapper

wrapper = TauAirlineEnvWrapper(max_steps=30)
transition = wrapper.reset(task_id="airline-task-001")

while not transition["done"]:
    action = {"tool": "some_airline_api", "arguments": {}}
    transition = wrapper.step(action)
```

### 奖励计算

```python
from src.algorithms.reward_model import GRPORewardModel, ToolSpec

reward_model = GRPORewardModel(
    tool_specs={
        "search_flight": ToolSpec(
            required_params={"origin", "destination"},
            optional_params={"date"},
        )
    }
)

result = reward_model.compute_reward(
    output="""
Thought: 用户需要从 SFO 到 JFK 的航班。
Action: search_flight(origin='SFO', destination='JFK')
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

## 研发路线图

- [x] 初始化项目目录与依赖声明。
- [x] 实现 tau-bench Airline 环境 veRL 适配层。
- [x] 实现 PRM-Lite / LATA / Group Advantage 奖励模块。
- [x] 添加核心单元测试。
- [ ] 接入 veRL rollout worker 与数据结构。
- [ ] 实现 GRPO 训练入口脚本。
- [ ] 添加 Hydra/OmegaConf 实验配置。
- [ ] 接入真实 tau-bench Airline 评测流程。
- [ ] 输出 baseline 与 ablation 实验结果。

## 设计原则

1. **长程任务优先**：面向真实业务中的多步工具调用，而非只优化短问答。
2. **过程监督优先**：通过 PRM-Lite 提供更密集的中间反馈，降低稀疏终局奖励训练难度。
3. **工程可复现**：所有核心模块配套测试，训练配置与实验流程尽可能标准化。
4. **可扩展环境接口**：先支持 tau-bench Airline，后续可扩展至 retail、web task、database task 等更多工具环境。

## License

本项目当前以研究与工程实验为目标。具体开源协议请以仓库中的 LICENSE 文件为准；如尚未补充 LICENSE，请在正式发布前完善。
