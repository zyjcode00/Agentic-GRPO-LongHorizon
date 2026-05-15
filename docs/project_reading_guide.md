# Agentic-GRPO-LongHorizon 项目阅读指南

> 本文档用于帮助你对照代码逐步理解 `Agentic-GRPO-LongHorizon` 项目。建议阅读时一边打开对应源码文件，一边按照本文的主线从入口、采样、奖励、训练、环境和 token mask 六个部分阅读。

---

## 1. 项目一句话概括

这个项目是一个面向 **长程工具调用 Agent** 的强化学习训练框架。

它的目标是训练一个大语言模型 Agent，让它在类似航空客服这样的多步骤任务中能够：

1. 理解用户需求；
2. 进行中间推理；
3. 调用工具/API；
4. 根据工具返回继续决策；
5. 最终完成任务；
6. 使用 GRPO 强化学习方法优化模型行为。

项目核心关键词：

- **tau-bench Airline**：航空客服类长程任务环境；
- **vLLM Rollout**：用大模型并发生成回答或行动轨迹；
- **PRM-Lite Reward Model**：对中间推理、工具调用和最终结果打分；
- **LATA**：长度归一化，避免模型靠冗长输出刷奖励；
- **GRPO**：对同一 prompt 的多条回答做组内 advantage 归一化；
- **Trainer**：把 rollout 样本变成训练样本，并执行训练更新。

---

## 2. 推荐阅读顺序

建议不要一开始就逐行阅读所有文件，而是按下面顺序理解：

```text
1. scripts/train_agentic_grpo.py      # 项目入口与训练主循环
2. src/models/vllm_adapter.py         # 模型 rollout / vLLM 采样
3. src/algorithms/reward_model.py     # 奖励模型：如何判断回答好不好
4. src/algorithms/grpo_trainer.py     # GRPO advantage、buffer、训练逻辑
5. envs/tau_env_wrapper.py            # tau-bench 航空环境包装
6. src/utils/token_utils.py            # response/thought/action mask 构造
7. configs/grpo_config.yaml            # 配置如何驱动上述模块
```

理解主线时，优先抓住这条数据流：

```text
Prompt
  ↓
AsyncVLLMRollout.generate_iter()
  ↓
VLLMGeneration
  ↓
generate_and_score()
  ↓
GRPOTrainer.ingest_rollouts()
  ↓
GRPORewardModel.compute_reward()
  ↓
compute_advantages_distributed()
  ↓
DistributedGRPOBuffer.add()
  ↓
GRPOSample
  ↓
GRPOTrainer.train_batch()
```

---

## 3. 项目目录架构

项目可以分成五个主要层次：

```text
Agentic-GRPO-LongHorizon/
├── configs/
│   └── grpo_config.yaml
├── envs/
│   └── tau_env_wrapper.py
├── scripts/
│   └── train_agentic_grpo.py
├── src/
│   ├── algorithms/
│   │   ├── reward_model.py
│   │   └── grpo_trainer.py
│   ├── models/
│   │   └── vllm_adapter.py
│   └── utils/
│       └── token_utils.py
└── tests/
```

| 层次 | 文件 | 主要作用 |
|---|---|---|
| 入口层 | `scripts/train_agentic_grpo.py` | 启动训练，加载配置，串起 rollout、reward、trainer |
| 采样层 | `src/models/vllm_adapter.py` | 调用 vLLM 异步生成模型回答 |
| 奖励层 | `src/algorithms/reward_model.py` | 解析模型输出，对推理、工具调用、最终结果打分 |
| 训练层 | `src/algorithms/grpo_trainer.py` | 计算 GRPO advantage，管理 buffer，执行训练 |
| 环境层 | `envs/tau_env_wrapper.py` | 包装 tau-bench Airline 环境 |
| Token 工具层 | `src/utils/token_utils.py` | 构造 response/thought/action 等训练 mask |
| 配置层 | `configs/grpo_config.yaml` | 定义训练、奖励、工具规范等参数 |

---

## 4. 总体运行流程

从入口来看，一次训练大致如下：

```text
加载 YAML 配置
  ↓
构造 AsyncVLLMRollout
  ↓
构造 GRPOTrainer
  ↓
选取一批 prompt
  ↓
调用 vLLM 生成 response
  ↓
奖励模型给 response 打分
  ↓
计算 GRPO advantage
  ↓
样本写入 buffer
  ↓
trainer.train_batch() 执行训练更新
  ↓
打印指标
  ↓
按需保存 checkpoint 和执行 unseen evaluation
```

核心心智图：

```text
scripts/train_agentic_grpo.py
        |
        v
AsyncVLLMRollout  ----调用----> vLLM / fake engine
        |
        v
VLLMGeneration
        |
        v
generate_and_score()
        |
        v
GRPOTrainer.ingest_rollouts()
        |
        +----> GRPORewardModel.compute_reward()
        |
        +----> compute_group_advantages()
        |
        v
DistributedGRPOBuffer
        |
        v
GRPOTrainer.train_batch()
```

环境和 token mask 是两条辅助线：

```text
TauAirlineEnvWrapper：负责真实任务环境交互
token_utils.py：负责训练时 token mask 对齐
```

---

## 5. 入口层：`scripts/train_agentic_grpo.py`

### 5.1 这个文件负责什么？

这是项目的训练入口，主要职责是：

1. 读取配置；
2. 初始化 rollout 引擎；
3. 初始化 trainer；
4. 准备训练 prompt；
5. 执行训练循环；
6. 输出训练指标；
7. 保存 checkpoint；
8. 进行可选评估。

你可以把它理解成项目的“总调度器”。

### 5.2 核心类

重点看：

```python
AgenticGRPOEntrypoint
```

这个类通常会封装：

- 配置对象；
- rollout 对象；
- trainer 对象；
- prompt 数据；
- 训练状态。

### 5.3 核心方法：`train_loop()`

主循环逻辑大致是：

```python
async def train_loop(self):
    for step in range(1, total_steps + 1):
        batch_prompts = select_prompt_batch(...)
        group_ids = [...]
        contexts = [...]
        env_session_ids = [...]

        rollout_batch = await generate_and_score(
            self.rollout,
            batch_prompts,
            trainer=self.trainer,
            ...
        )

        train_metrics = self.trainer.train_batch()
```

这段代码是理解项目的最小主线。

阅读时建议重点回答三个问题：

1. `batch_prompts` 从哪里来？
2. `generate_and_score()` 返回了什么？
3. `trainer.train_batch()` 消费的样本是在哪里写入的？

---

## 6. 采样层：`src/models/vllm_adapter.py`

### 6.1 文件作用

这个文件负责把 vLLM 的异步生成接口包装成项目内部统一格式。

核心类：

```python
class AsyncVLLMRollout:
```

它负责：

- 调用 vLLM engine；
- 支持异步并发生成；
- 控制最大并发数；
- 收集 token 数、耗时、TPS 等指标；
- 把原始输出解析成统一结构 `VLLMGeneration`。

### 6.2 标准输出结构：`VLLMGeneration`

模型生成结果会被统一成：

```python
@dataclass(frozen=True)
class VLLMGeneration:
    prompt: str
    response: str
    logprobs: Any = None
    token_ids: Sequence[int] = field(default_factory=tuple)
    request_id: str = ""
    group_id: Any | None = None
    context: Mapping[str, Any] = field(default_factory=dict)
    env_session_id: str | None = None
    latency_s: float = 0.0
```

这个结构说明，项目内部并不直接到处传 vLLM 原始返回，而是传一个标准化对象。

最重要字段：

| 字段 | 含义 |
|---|---|
| `prompt` | 输入问题或任务 |
| `response` | 模型生成内容 |
| `logprobs` | token log probability，用于训练或分析 |
| `token_ids` | 输出 token id |
| `group_id` | GRPO 组内比较使用的分组 id |
| `context` | 环境、工具调用等上下文信息 |
| `env_session_id` | 当前环境会话 id |
| `latency_s` | 单次生成耗时 |

### 6.3 为什么使用 `ContextVar`？

该文件中使用了类似下面的上下文变量：

```python
_env_session_id = contextvars.ContextVar(...)
_tool_context = contextvars.ContextVar(...)
```

原因是 rollout 是异步并发的。

假设同时有多个航空客服任务：

```text
任务 A：用户想改签
任务 B：用户想查航班
任务 C：用户想退票
```

每个任务都有独立的环境状态和工具上下文。如果用普通全局变量，很容易把 A 的状态传给 B。`ContextVar` 可以保证每个 coroutine 有自己的上下文。

### 6.4 `generate_iter()`

`generate_iter()` 是异步生成的主接口。它会：

1. 接收一批 prompts；
2. 为每个 prompt 创建异步任务；
3. 并发调用 `_generate_one()`；
4. 谁先生成完，谁先 yield；
5. 记录 TPS 等指标；
6. 最终返回 `(index, VLLMGeneration)`。

它返回 index 的原因是：异步任务完成顺序不一定等于输入顺序，需要靠 index 重新对齐。

### 6.5 `_generate_one()`

`_generate_one()` 负责单条样本的生成：

```text
构造 request_id
  ↓
绑定 rollout context
  ↓
调用 vLLM engine
  ↓
解析 raw output
  ↓
合并 context
  ↓
返回 VLLMGeneration
```

### 6.6 `generate_and_score()`

这个函数非常关键，它连接采样层和训练层。

它做两件事：

第一，调用 rollout 生成：

```python
async for index, generation in rollout.generate_iter(...):
    ...
```

第二，把最终结果交给 trainer：

```python
samples = trainer.ingest_rollouts(
    prompts_out,
    responses,
    logprobs,
    group_ids=group_ids_out,
    contexts=contexts_out,
    outcome_rewards=outcome_rewards,
    gather_fn=gather_fn,
)
```

所以它是：

```text
vLLM 世界 → GRPO Trainer 世界
```

的桥梁。

---

## 7. 奖励层：`src/algorithms/reward_model.py`

### 7.1 文件作用

这个文件定义模型回答如何被打分。

核心类：

```python
class GRPORewardModel:
```

它通常会处理：

1. 解析模型输出中的 `Thought`、`Action`、`Observation`、`Final`；
2. 判断工具调用是否合法；
3. 判断推理是否有依据；
4. 判断是否重复、臆造、缺少参数；
5. 综合过程奖励和最终奖励；
6. 使用长度归一化得到最终 reward。

### 7.2 模型输出格式

项目倾向于让模型输出类似结构：

```text
Thought: 用户想从 SFO 到 JFK，需要先查询航班。
Action: search_flights(origin='SFO', destination='JFK')
Observation: 找到 3 个可选航班。
Final: 已为用户提供可选航班。
```

这些字段的大致含义：

| 字段 | 含义 |
|---|---|
| `Thought` | 模型中间推理 |
| `Action` | 工具/API 调用 |
| `Observation` | 工具返回或环境反馈 |
| `Final` | 最终答复 |

### 7.3 工具调用奖励

项目会根据配置中的工具规范检查工具调用。

例如配置中可能有：

```yaml
tool_specs:
  search_flights:
    required_params: [origin, destination]
    optional_params: [date, cabin]
  book_reservation:
    required_params: [flight_id, passenger_name]
    optional_params: [seat]
```

如果模型输出：

```text
Action: search_flights(origin='SFO', destination='JFK')
```

这通常是合法工具调用。

如果模型输出：

```text
Action: search_flights(origin='SFO')
```

少了必需参数 `destination`，就会被扣分。

### 7.4 过程奖励示例

奖励模型会鼓励：

| 行为 | 倾向 |
|---|---|
| 调用存在的工具 | 加分 |
| 工具参数完整 | 加分 |
| 推理和 prompt 相关 | 加分 |
| 正确使用 observation | 加分 |
| 最终回答完成任务 | 加分 |

奖励模型会惩罚：

| 行为 | 倾向 |
|---|---|
| 调用不存在的工具 | 扣分 |
| 缺少必需参数 | 扣分 |
| 编造 observation 没有的信息 | 扣分 |
| 重复推理 | 扣分 |
| 输出过长且无效 | 扣分 |

### 7.5 LATA 长度归一化

项目中一个重要设计是长度归一化。

直觉是：不能让模型通过写很长的废话拿高分。

形式大致是：

```text
final_reward = raw_reward / sqrt(length)
```

例如：

```text
短回答 raw_reward = 1.0, length = 4
长回答 raw_reward = 1.0, length = 100
```

归一化后：

```text
短回答 final_reward = 1.0 / sqrt(4) = 0.5
长回答 final_reward = 1.0 / sqrt(100) = 0.1
```

因此项目鼓励的是：

```text
有效、简洁、工具调用正确
```

而不是：

```text
冗长、重复、看起来很努力
```

---

## 8. 训练层：`src/algorithms/grpo_trainer.py`

### 8.1 文件作用

这个文件负责把 rollout 结果转换成强化学习训练样本。

核心类通常包括：

```python
GRPOTrainer
DistributedGRPOBuffer
GRPOSample
```

核心函数通常包括：

```python
compute_group_advantages()
compute_advantages_distributed()
```

### 8.2 `GRPOSample`

训练样本通常会被组织成：

```python
@dataclass(frozen=True)
class GRPOSample:
    prompt: str
    response: str
    logprobs: Any
    advantage: float
    group_id: Any | None = None
    reward: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

字段含义：

| 字段 | 含义 |
|---|---|
| `prompt` | 输入任务 |
| `response` | 模型输出 |
| `logprobs` | 模型生成时的 log probability |
| `advantage` | GRPO 计算出的相对优势 |
| `group_id` | 同一 prompt 或同一任务组的 id |
| `reward` | 奖励模型给出的原始或最终分数 |
| `metadata` | 额外信息，例如环境、工具调用、mask 等 |

### 8.3 `ingest_rollouts()`

`ingest_rollouts()` 是采样结果进入训练系统的关键入口。

它大致做：

```text
接收 prompts/responses/logprobs/group_ids/contexts
  ↓
调用 reward model 打分
  ↓
按 group_id 计算 advantage
  ↓
构造 GRPOSample
  ↓
写入 DistributedGRPOBuffer
```

这一步非常关键，因为它把普通模型输出变成了训练样本。

### 8.4 GRPO 的核心直觉

GRPO 关注的是：

```text
对于同一个 prompt，多个回答里，哪个相对更好？
```

例如同一问题采样 4 个回答：

```text
A reward = 0.2
B reward = 0.5
C reward = 0.1
D reward = 0.8
```

先计算组内均值和标准差，然后得到：

```text
advantage = (reward - group_mean) / group_std
```

因此：

- D 的 advantage 较高，训练会鼓励模型更可能生成类似回答；
- C 的 advantage 较低，训练会降低模型生成类似回答的概率。

它不是只看绝对分数，而是看组内相对好坏。

### 8.5 `train_batch()`

`train_batch()` 会从 buffer 中取样本并执行训练更新。

阅读时建议关注：

1. 它从哪里取样本？
2. 它如何使用 `advantage`？
3. 它是否真正调用了底层模型训练接口，还是当前实现中存在 mock/fallback？
4. 它返回哪些训练指标？

---

## 9. 环境层：`envs/tau_env_wrapper.py`

### 9.1 文件作用

这个文件把 tau-bench 的 Airline 环境包装成项目内部更易使用的接口。

核心类：

```python
class TauAirlineEnvWrapper:
```

它通常会提供类似：

```python
reset()
step(action)
```

并返回标准结构：

```python
{
    "observation": ...,
    "reward": float,
    "done": bool,
    "info": dict,
}
```

### 9.2 为什么要包装环境？

因为外部环境的 API 不一定和本项目训练/采样逻辑一致。

包装层的作用是：

```text
tau-bench 原始环境
  ↓
TauAirlineEnvWrapper
  ↓
项目内部统一环境接口
```

这样 trainer 和 rollout 不需要关心 tau-bench 的底层细节。

### 9.3 并发状态隔离

类似 vLLM adapter，环境包装也可能使用 `ContextVar` 保存当前环境状态。

原因是多个 rollout 可能同时运行：

```text
rollout A：会话 A 的环境状态
rollout B：会话 B 的环境状态
rollout C：会话 C 的环境状态
```

如果状态混在一起，就会导致错误的工具反馈或奖励计算。

---

## 10. Token 工具层：`src/utils/token_utils.py`

### 10.1 文件作用

这个文件负责构造训练时需要的 token mask。

核心函数：

```python
get_grpo_masks()
```

常见 mask 包括：

```text
attention_mask
response_mask
thought_mask
action_mask
```

### 10.2 为什么需要 mask？

训练时不是所有 token 都应该同等参与 loss。

例如：

```text
用户 prompt：通常不作为模型需要学习生成的目标
assistant response：通常是优化重点
Thought 部分：可能单独分析或加权
Action 部分：工具调用，可能需要特别监督
Final 部分：最终答复
```

因此需要 mask 标记不同区域。

### 10.3 Render-Twice-Diff 方法

项目使用的思路可以理解为：

```text
第一次渲染：只渲染用户 prompt
第二次渲染：渲染用户 prompt + assistant response
对比两次结果，找出 response token 起点
```

这能较准确地区分 prompt token 和 response token。

---

## 11. 配置层：`configs/grpo_config.yaml`

配置文件驱动训练、奖励和工具规范。

重点关注几个区域：

### 11.1 训练相关

可能包括：

```yaml
training:
  total_steps: ...
  batch_size: ...
  group_size: ...
```

这些参数决定：

- 总训练步数；
- 每步采样多少 prompt；
- 每个 prompt 采样几个回答；
- 多久保存 checkpoint；
- 多久评估一次。

### 11.2 Rollout 相关

可能包括：

```yaml
rollout:
  model: ...
  temperature: ...
  max_tokens: ...
  max_concurrency: ...
```

这些参数决定模型生成行为。

### 11.3 奖励相关

例如：

```yaml
reward_model:
  correct_tool_reward: 0.4
  invalid_tool_penalty: -0.6
  reasoning_reward: 0.15
  unsupported_reasoning_penalty: -0.1
  observation_reward: 0.05
  repetition_penalty: -0.2
```

这些配置会影响模型偏好。

### 11.4 工具规范

例如：

```yaml
tool_specs:
  search_flights:
    required_params: [origin, destination]
    optional_params: [date, cabin]
  book_reservation:
    required_params: [flight_id, passenger_name]
    optional_params: [seat]
```

奖励模型会用这些规范判断工具调用是否合法。

---

## 12. 一个具体例子：哪种回答更可能得高分？

### 回答 A

```text
Thought: 用户想从 SFO 到 JFK，需要先查询航班。
Action: search_flights(origin='SFO', destination='JFK')
Observation: 找到 3 个可选航班。
Final: 已为用户提供可选航班。
```

### 回答 B

```text
Thought: 我觉得用户可能想去纽约，也许今天机票很多。我将直接确认。
Final: 已经为你成功预订了最好的航班。
```

按照项目奖励设计，回答 A 更可能得高分。

原因：

1. A 有明确的 `Thought → Action → Observation → Final` 结构；
2. A 调用了合适的工具 `search_flights`；
3. A 的工具参数包含 `origin` 和 `destination`；
4. A 的最终回答基于工具观察结果；
5. B 没有调用工具，却声称已经预订成功；
6. B 可能存在臆造结果的问题；
7. B 缺少可验证的中间步骤。

这个例子体现了项目想训练的 Agent 类型：

```text
不是直接瞎猜最终答案，而是通过合理推理和工具调用逐步完成任务。
```

---

## 13. 对照代码阅读路线

### 第一步：读入口文件

文件：

```text
scripts/train_agentic_grpo.py
```

重点找：

```python
AgenticGRPOEntrypoint
train_loop
main
```

读完后你应该知道：

- 配置如何加载；
- rollout 如何初始化；
- trainer 如何初始化；
- 每个 step 做什么。

### 第二步：读采样文件

文件：

```text
src/models/vllm_adapter.py
```

重点找：

```python
AsyncVLLMRollout
VLLMGeneration
generate_iter
_generate_one
generate_and_score
```

读完后你应该知道：

- prompt 如何送给 vLLM；
- 异步结果如何对齐顺序；
- 生成结果如何标准化；
- 生成结果如何进入 trainer。

### 第三步：读奖励文件

文件：

```text
src/algorithms/reward_model.py
```

重点找：

```python
GRPORewardModel
compute_reward
score_response
score_batch
```

读完后你应该知道：

- 模型输出如何解析；
- 工具调用如何检查；
- 推理和观察如何奖励；
- 长度归一化如何生效。

### 第四步：读训练文件

文件：

```text
src/algorithms/grpo_trainer.py
```

重点找：

```python
GRPOTrainer
GRPOSample
DistributedGRPOBuffer
compute_group_advantages
compute_advantages_distributed
ingest_rollouts
train_batch
```

读完后你应该知道：

- reward 如何变成 advantage；
- 样本如何进入 buffer；
- trainer 如何消费样本；
- 训练指标如何产生。

### 第五步：读环境包装

文件：

```text
envs/tau_env_wrapper.py
```

重点找：

```python
TauAirlineEnvWrapper
reset
step
```

读完后你应该知道：

- tau-bench 环境如何接入；
- observation/reward/done/info 如何统一；
- 并发环境状态如何隔离。

### 第六步：读 token mask 工具

文件：

```text
src/utils/token_utils.py
```

重点找：

```python
get_grpo_masks
```

读完后你应该知道：

- response mask 如何定位；
- thought/action mask 如何区分；
- 为什么 prompt token 和 response token 要分开。

---

## 14. 最后总结

这个项目的核心不是单纯调用 LLM，也不是单纯写一个 reward function，而是把下面几件事串成一个训练闭环：

```text
大模型生成
  ↓
工具调用轨迹
  ↓
过程奖励和结果奖励
  ↓
组内相对 advantage
  ↓
强化学习训练
  ↓
更好的 Agent 行为
```

如果只记一条主线，请记住：

```text
train_agentic_grpo.py
  → vllm_adapter.py
  → reward_model.py
  → grpo_trainer.py
```

如果只记一个目标，请记住：

```text
训练一个能在长程任务中合理推理、正确调用工具、简洁完成任务的 Agent。
```
