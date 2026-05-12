"""Reward utilities for GRPO training.

This module implements three pieces used by the project training loop:

* PRM-Lite: a lightweight process reward model that scores intermediate
  reasoning/tool-use steps instead of only scoring the final answer.
* LATA: length-aware reward normalisation, ``reward / sqrt(length)``.
* Group Advantage: per-prompt normalisation for the N responses sampled by GRPO.

The implementation is intentionally dependency-free so it can run inside rollout
workers and unit tests without requiring a heavyweight reward model service.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


ToolValidator = Callable[[Mapping[str, Any], Mapping[str, Any]], bool]


@dataclass(frozen=True)
class ToolSpec:
    """Validation rules for a callable tool/API.

    Attributes:
        required_params: Parameters that must be present in a tool call.
        optional_params: Parameters that are allowed but not required. If both
            ``required_params`` and ``optional_params`` are empty, any parameter
            name is accepted.
        validators: Optional per-parameter predicates. A predicate receives the
            parameter value and returns ``True`` when it is legal.
    """

    required_params: set[str] = field(default_factory=set)
    optional_params: set[str] = field(default_factory=set)
    validators: Mapping[str, Callable[[Any], bool]] = field(default_factory=dict)

    @property
    def allowed_params(self) -> set[str]:
        return set(self.required_params) | set(self.optional_params)


@dataclass(frozen=True)
class ThoughtStep:
    """One parsed step from a model thought chain."""

    index: int
    kind: str
    content: str
    tool_name: str | None = None
    tool_args: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepReward:
    """Reward attribution for one PRM-Lite step."""

    step_index: int
    kind: str
    reward: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RewardBreakdown:
    """Detailed reward result for a single sampled response."""

    raw_reward: float
    final_reward: float
    length: int
    step_rewards: tuple[StepReward, ...]
    outcome_reward: float = 0.0


class GRPORewardModel:
    """Lightweight GRPO reward model with PRM-Lite, LATA and group advantage.

    Parameters are deliberately simple and can be tuned by the trainer config.
    The default values favour valid, evidence-grounded tool use while penalising
    malformed API calls, unsupported tools and verbose filler reasoning.
    """

    _STEP_MARKER = re.compile(
        r"(?im)^\s*(?:step\s*\d+|thought|action|observation|final)\s*[:：]"
    )
    _TOOL_CALL_PATTERNS = (
        re.compile(r"(?is)<tool_call>\s*(.*?)\s*</tool_call>"),
        re.compile(r"(?im)^\s*Action\s*[:：]\s*(.+)$"),
        re.compile(r"(?im)^\s*Tool\s*[:：]\s*(.+)$"),
    )
    _WORD_RE = re.compile(r"\w+", re.UNICODE)

    def __init__(
        self,
        tool_specs: Mapping[str, ToolSpec | Mapping[str, Any]] | None = None,
        *,
        correct_tool_reward: float = 0.4,
        invalid_tool_penalty: float = -0.6,
        reasoning_reward: float = 0.15,
        unsupported_reasoning_penalty: float = -0.1,
        observation_reward: float = 0.05,
        repetition_penalty: float = -0.2,
        max_step_reward: float = 1.0,
        length_normalization: bool = True,
        min_length: int = 1,
        std_epsilon: float = 1e-8,
    ) -> None:
        self.tool_specs = self._normalise_tool_specs(tool_specs or {})
        self.correct_tool_reward = correct_tool_reward
        self.invalid_tool_penalty = invalid_tool_penalty
        self.reasoning_reward = reasoning_reward
        self.unsupported_reasoning_penalty = unsupported_reasoning_penalty
        self.observation_reward = observation_reward
        self.repetition_penalty = repetition_penalty
        self.max_step_reward = max_step_reward
        self.length_normalization = length_normalization
        self.min_length = min_length
        self.std_epsilon = std_epsilon

    def compute_reward(
        self,
        output: str,
        *,
        prompt: str = "",
        context: Mapping[str, Any] | None = None,
        outcome_reward: float = 0.0,
    ) -> RewardBreakdown:
        """Compute PRM-Lite raw reward and LATA-normalised final reward.

        Args:
            output: Raw model response containing thought/action/observation text.
            prompt: Original prompt. It is used as evidence for simple reasoning
                consistency checks.
            context: Optional rollout state. Supported keys include
                ``known_facts`` (strings the model may cite), ``tool_specs``
                (extra/overriding tool specs), and ``tool_validators`` mapping
                tool name to a custom ``(args, context) -> bool`` validator.
            outcome_reward: Optional terminal/result reward supplied by the
                environment or evaluator. It is added to the process reward.
        """

        context = dict(context or {})
        steps = self.parse_thought_chain(output)
        step_rewards = tuple(
            self._score_step(step, prompt=prompt, previous_steps=steps[:idx], context=context)
            for idx, step in enumerate(steps)
        )
        process_reward = sum(step.reward for step in step_rewards)
        raw_reward = process_reward + float(outcome_reward)
        length = self.measure_length(output)
        final_reward = self.apply_lata(raw_reward, length) if self.length_normalization else raw_reward
        return RewardBreakdown(
            raw_reward=raw_reward,
            final_reward=final_reward,
            length=length,
            step_rewards=step_rewards,
            outcome_reward=float(outcome_reward),
        )

    def compute_group_advantages(
        self,
        rewards: Sequence[float | RewardBreakdown],
        group_ids: Sequence[Any] | None = None,
    ) -> list[float]:
        """Return GRPO mean/std normalised advantages.

        ``rewards`` may be floats or ``RewardBreakdown`` instances. When
        ``group_ids`` is provided, normalisation is performed independently for
        each prompt group. A zero-variance group receives all-zero advantages.
        """

        values = [r.final_reward if isinstance(r, RewardBreakdown) else float(r) for r in rewards]
        if group_ids is None:
            group_ids = [0] * len(values)
        if len(group_ids) != len(values):
            raise ValueError("group_ids must have the same length as rewards")

        grouped: dict[Any, list[int]] = {}
        for idx, gid in enumerate(group_ids):
            grouped.setdefault(gid, []).append(idx)

        advantages = [0.0] * len(values)
        for indices in grouped.values():
            group_values = [values[i] for i in indices]
            mean = sum(group_values) / len(group_values)
            variance = sum((v - mean) ** 2 for v in group_values) / len(group_values)
            std = math.sqrt(variance)
            if std <= self.std_epsilon:
                continue
            for i in indices:
                advantages[i] = (values[i] - mean) / (std + self.std_epsilon)
        return advantages

    def parse_thought_chain(self, output: str) -> tuple[ThoughtStep, ...]:
        """Parse thought/action/observation/final steps from model output."""

        chunks = self._split_into_chunks(output)
        steps: list[ThoughtStep] = []
        for raw_chunk in chunks:
            chunk = raw_chunk.strip()
            if not chunk:
                continue
            kind = self._infer_kind(chunk)
            tool_name = None
            tool_args: Mapping[str, Any] = {}
            if kind == "tool_call":
                tool_name, tool_args = self._parse_tool_call(chunk)
            steps.append(
                ThoughtStep(
                    index=len(steps),
                    kind=kind,
                    content=chunk,
                    tool_name=tool_name,
                    tool_args=tool_args,
                )
            )
        if not steps and output.strip():
            steps.append(ThoughtStep(index=0, kind="thought", content=output.strip()))
        return tuple(steps)

    def apply_lata(self, raw_reward: float, length: int | None = None, output: str | None = None) -> float:
        """Apply Length-Aware Training normalisation: reward / sqrt(length)."""

        if length is None:
            if output is None:
                raise ValueError("Either length or output must be provided")
            length = self.measure_length(output)
        effective_length = max(int(length), self.min_length)
        return float(raw_reward) / math.sqrt(effective_length)

    def measure_length(self, output: str) -> int:
        """Measure response length in word-like tokens for LATA."""

        return max(len(self._WORD_RE.findall(output)), self.min_length)

    def _score_step(
        self,
        step: ThoughtStep,
        *,
        prompt: str,
        previous_steps: Sequence[ThoughtStep],
        context: Mapping[str, Any],
    ) -> StepReward:
        reward = 0.0
        reasons: list[str] = []

        if self._is_repetitive(step, previous_steps):
            reward += self.repetition_penalty
            reasons.append("repetitive_step")

        if step.kind == "tool_call":
            ok, tool_reasons = self._validate_tool_call(step, context)
            if ok:
                reward += self.correct_tool_reward
                reasons.extend(tool_reasons or ["valid_tool_call"])
            else:
                reward += self.invalid_tool_penalty
                reasons.extend(tool_reasons or ["invalid_tool_call"])
        elif step.kind == "thought":
            if self._is_reasoning_consistent(step.content, prompt, previous_steps, context):
                reward += self.reasoning_reward
                reasons.append("evidence_grounded_reasoning")
            else:
                reward += self.unsupported_reasoning_penalty
                reasons.append("unsupported_reasoning")
        elif step.kind == "observation":
            reward += self.observation_reward
            reasons.append("observation_integrated")
        elif step.kind == "final":
            if self._is_reasoning_consistent(step.content, prompt, previous_steps, context):
                reward += self.reasoning_reward
                reasons.append("grounded_final_answer")
            else:
                reasons.append("final_answer")

        reward = max(-self.max_step_reward, min(self.max_step_reward, reward))
        return StepReward(step.index, step.kind, reward, tuple(reasons))

    def _validate_tool_call(
        self, step: ThoughtStep, context: Mapping[str, Any]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if not step.tool_name:
            return False, ["missing_tool_name"]

        tool_specs = dict(self.tool_specs)
        tool_specs.update(self._normalise_tool_specs(context.get("tool_specs", {})))
        spec = tool_specs.get(step.tool_name)
        if spec is None and tool_specs:
            return False, ["unsupported_tool"]

        args = dict(step.tool_args)
        if spec is not None:
            missing = sorted(spec.required_params - set(args))
            if missing:
                reasons.append(f"missing_required_params:{','.join(missing)}")
            allowed = spec.allowed_params
            if allowed:
                extra = sorted(set(args) - allowed)
                if extra:
                    reasons.append(f"unknown_params:{','.join(extra)}")
            for name, validator in spec.validators.items():
                if name in args and not validator(args[name]):
                    reasons.append(f"invalid_param:{name}")

        validators = context.get("tool_validators", {}) or {}
        custom_validator = validators.get(step.tool_name) if isinstance(validators, Mapping) else None
        if custom_validator is not None and not custom_validator(args, context):
            reasons.append("custom_validator_failed")

        return not reasons, reasons or ["valid_tool_call"]

    def _is_reasoning_consistent(
        self,
        content: str,
        prompt: str,
        previous_steps: Sequence[ThoughtStep],
        context: Mapping[str, Any],
    ) -> bool:
        text = content.lower()
        evidence_parts = [prompt]
        evidence_parts.extend(step.content for step in previous_steps if step.kind in {"observation", "tool_call"})
        evidence_parts.extend(str(fact) for fact in context.get("known_facts", []) or [])
        evidence = "\n".join(evidence_parts).lower()

        referenced_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
        if referenced_numbers and not all(number in evidence for number in referenced_numbers):
            return False

        meaningful_tokens = {
            token
            for token in self._WORD_RE.findall(text)
            if len(token) > 3 and token not in {"therefore", "because", "should", "would", "could"}
        }
        if not meaningful_tokens:
            return False
        evidence_tokens = set(self._WORD_RE.findall(evidence))
        overlap = meaningful_tokens & evidence_tokens
        has_reasoning_cue = any(cue in text for cue in ("because", "therefore", "so ", "since", "based on"))
        return bool(overlap) or has_reasoning_cue

    def _is_repetitive(self, step: ThoughtStep, previous_steps: Sequence[ThoughtStep]) -> bool:
        current = " ".join(self._WORD_RE.findall(step.content.lower()))
        if not current:
            return False
        return any(current == " ".join(self._WORD_RE.findall(prev.content.lower())) for prev in previous_steps)

    def _split_into_chunks(self, output: str) -> list[str]:
        matches = list(self._STEP_MARKER.finditer(output))
        if not matches:
            return [part for part in re.split(r"\n\s*\n", output) if part.strip()]
        chunks: list[str] = []
        for idx, match in enumerate(matches):
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(output)
            chunks.append(output[match.start() : end])
        prefix = output[: matches[0].start()].strip()
        if prefix:
            chunks.insert(0, prefix)
        return chunks

    def _infer_kind(self, chunk: str) -> str:
        lower = chunk.lower().lstrip()
        if lower.startswith("observation"):
            return "observation"
        if lower.startswith("final"):
            return "final"
        if lower.startswith("action") or lower.startswith("tool") or "<tool_call>" in lower:
            return "tool_call"
        for pattern in self._TOOL_CALL_PATTERNS:
            if pattern.search(chunk):
                return "tool_call"
        return "thought"

    def _parse_tool_call(self, chunk: str) -> tuple[str | None, Mapping[str, Any]]:
        payload = chunk
        for pattern in self._TOOL_CALL_PATTERNS:
            match = pattern.search(chunk)
            if match:
                payload = match.group(1).strip()
                break
        payload = re.sub(r"^(?:Action|Tool)\s*[:：]\s*", "", payload.strip(), flags=re.I)

        parsed = self._parse_mapping(payload)
        if isinstance(parsed, Mapping):
            name = parsed.get("tool") or parsed.get("name") or parsed.get("api")
            args = parsed.get("arguments") or parsed.get("args") or parsed.get("parameters") or {}
            if name is None and len(parsed) == 1:
                name, args = next(iter(parsed.items()))
            return str(name) if name is not None else None, args if isinstance(args, Mapping) else {}

        call_match = re.match(r"([A-Za-z_]\w*)\s*\((.*)\)\s*$", payload, flags=re.S)
        if call_match:
            name = call_match.group(1)
            args = self._parse_call_args(call_match.group(2))
            return name, args
        return None, {}

    def _parse_mapping(self, payload: str) -> Any:
        try:
            return json.loads(payload)
        except Exception:
            pass
        try:
            return ast.literal_eval(payload)
        except Exception:
            return None

    def _parse_call_args(self, args_text: str) -> Mapping[str, Any]:
        if not args_text.strip():
            return {}
        try:
            expr = ast.parse(f"f({args_text})", mode="eval").body
            if isinstance(expr, ast.Call):
                return {kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords if kw.arg is not None}
        except Exception:
            return {}
        return {}

    def _normalise_tool_specs(
        self, specs: Mapping[str, ToolSpec | Mapping[str, Any]]
    ) -> dict[str, ToolSpec]:
        normalised: dict[str, ToolSpec] = {}
        for name, spec in specs.items():
            if isinstance(spec, ToolSpec):
                normalised[name] = spec
            elif isinstance(spec, Mapping):
                normalised[name] = ToolSpec(
                    required_params=set(spec.get("required_params", spec.get("required", []))),
                    optional_params=set(spec.get("optional_params", spec.get("optional", []))),
                    validators=spec.get("validators", {}),
                )
            else:
                raise TypeError(f"Unsupported ToolSpec for {name!r}: {type(spec)!r}")
        return normalised


_DEFAULT_MODEL = GRPORewardModel()


def compute_reward(
    output: str,
    *,
    prompt: str = "",
    context: Mapping[str, Any] | None = None,
    outcome_reward: float = 0.0,
    tool_specs: Mapping[str, ToolSpec | Mapping[str, Any]] | None = None,
) -> RewardBreakdown:
    """Convenience wrapper for scoring a single response."""

    model = GRPORewardModel(tool_specs=tool_specs) if tool_specs is not None else _DEFAULT_MODEL
    return model.compute_reward(output, prompt=prompt, context=context, outcome_reward=outcome_reward)


def apply_lata(raw_reward: float, length: int) -> float:
    """Convenience wrapper for LATA reward normalisation."""

    return _DEFAULT_MODEL.apply_lata(raw_reward, length=length)


def compute_group_advantages(
    rewards: Sequence[float | RewardBreakdown], group_ids: Sequence[Any] | None = None
) -> list[float]:
    """Convenience wrapper for GRPO group advantage normalisation."""

    return _DEFAULT_MODEL.compute_group_advantages(rewards, group_ids=group_ids)
