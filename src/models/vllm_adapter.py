"""Async vLLM rollout adapter with ContextVar-isolated tau-bench state.

The module is intentionally importable without vLLM installed.  Production code
can construct :class:`AsyncVLLMRollout` with ``model=...``/``engine_args=...`` to
create a real ``vllm.AsyncLLMEngine`` lazily, while tests and CPU-only CI can
inject a fake engine that implements the same ``generate`` coroutine/async
iterator surface.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable, Mapping, MutableMapping, Sequence


_env_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("vllm_env_session_id", default=None)
_tool_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("vllm_tool_context", default={})


@dataclass(frozen=True)
class RolloutContext:
    """Per-coroutine rollout context for tau-bench/tool execution."""

    env_session_id: str
    tool_context: Mapping[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "env_session_id": self.env_session_id,
            "tool_context": dict(self.tool_context),
        }


@dataclass(frozen=True)
class VLLMGeneration:
    """Normalised output from one vLLM request."""

    prompt: str
    response: str
    logprobs: Any = None
    token_ids: Sequence[int] = field(default_factory=tuple)
    request_id: str = ""
    group_id: Any | None = None
    context: Mapping[str, Any] = field(default_factory=dict)
    env_session_id: str | None = None
    latency_s: float = 0.0

    @property
    def num_tokens(self) -> int:
        if self.token_ids:
            return len(self.token_ids)
        return len(self.response.split())

    def as_rollout_dict(self) -> dict[str, Any]:
        """Return a dictionary aligned with ``GRPOTrainer.ingest_rollouts`` fields."""

        return {
            "prompt": self.prompt,
            "response": self.response,
            "logprobs": self.logprobs,
            "group_id": self.group_id,
            "context": dict(self.context),
            "metadata": {
                "request_id": self.request_id,
                "env_session_id": self.env_session_id,
                "latency_s": self.latency_s,
                "num_tokens": self.num_tokens,
            },
        }


@dataclass(frozen=True)
class RolloutBatch:
    """Batch payload that can be passed directly to ``GRPOTrainer.ingest_rollouts``."""

    prompts: list[str]
    responses: list[str]
    logprobs: list[Any]
    group_ids: list[Any]
    contexts: list[Mapping[str, Any] | None]
    generations: list[VLLMGeneration]
    rewards: list[Any] = field(default_factory=list)
    samples: list[Any] = field(default_factory=list)
    tps: float = 0.0
    total_tokens: int = 0
    elapsed_s: float = 0.0


def get_current_env_session_id() -> str | None:
    """Return the env session id bound to the current asyncio task."""

    return _env_session_id.get()


def get_current_tool_context() -> dict[str, Any]:
    """Return a copy of the current task-local tool context."""

    return dict(_tool_context.get() or {})


@contextlib.contextmanager
def rollout_context(env_session_id: str, tool_context: Mapping[str, Any] | None = None):
    """Temporarily bind rollout ContextVars in the current context."""

    env_token = _env_session_id.set(env_session_id)
    tool_token = _tool_context.set(dict(tool_context or {}))
    try:
        yield RolloutContext(env_session_id=env_session_id, tool_context=dict(tool_context or {}))
    finally:
        _tool_context.reset(tool_token)
        _env_session_id.reset(env_token)


class AsyncVLLMRollout:
    """Async parallel sampler around ``vllm.AsyncLLMEngine``.

    Parameters
    ----------
    engine:
        Optional already-created engine.  It only needs a ``generate`` method,
        making dependency-injected tests possible without installing vLLM.
    model / engine_args:
        Used to lazily construct a real ``AsyncLLMEngine`` when ``engine`` is not
        provided.
    sampling_params:
        Default vLLM ``SamplingParams`` or a plain mapping accepted by vLLM.
    max_concurrency:
        Optional semaphore limiting simultaneous in-flight requests.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        engine: Any | None = None,
        engine_args: Mapping[str, Any] | None = None,
        sampling_params: Any | None = None,
        max_concurrency: int | None = None,
        request_id_prefix: str = "grpo-rollout",
        print_tps: bool = True,
    ) -> None:
        self.engine = engine if engine is not None else self._build_engine(model=model, engine_args=engine_args)
        self.sampling_params = self._build_sampling_params(sampling_params)
        self.max_concurrency = max_concurrency
        self.request_id_prefix = request_id_prefix
        self.print_tps = print_tps
        self.last_metrics: dict[str, float] = {}

    async def generate(
        self,
        prompts: Sequence[str],
        *,
        sampling_params: Any | None = None,
        contexts: Sequence[Mapping[str, Any] | None] | None = None,
        group_ids: Sequence[Any] | None = None,
        env_session_ids: Sequence[str | None] | None = None,
    ) -> list[VLLMGeneration]:
        """Generate responses for a batch of prompts concurrently."""

        indexed: list[tuple[int, VLLMGeneration]] = []
        async for index, generation in self.generate_iter(
            prompts,
            sampling_params=sampling_params,
            contexts=contexts,
            group_ids=group_ids,
            env_session_ids=env_session_ids,
        ):
            indexed.append((index, generation))
        indexed.sort(key=lambda item: item[0])
        return [generation for _, generation in indexed]

    async def generate_iter(
        self,
        prompts: Sequence[str],
        *,
        sampling_params: Any | None = None,
        contexts: Sequence[Mapping[str, Any] | None] | None = None,
        group_ids: Sequence[Any] | None = None,
        env_session_ids: Sequence[str | None] | None = None,
    ) -> AsyncIterator[tuple[int, VLLMGeneration]]:
        """Yield generations as soon as individual requests finish."""

        prompts = list(prompts)
        contexts = _normalise_optional_sequence(contexts, len(prompts), default=None, name="contexts")
        group_ids = list(group_ids) if group_ids is not None else list(prompts)
        env_session_ids = _normalise_optional_sequence(env_session_ids, len(prompts), default=None, name="env_session_ids")
        if len(group_ids) != len(prompts):
            raise ValueError("group_ids must have the same length as prompts")

        params = self._build_sampling_params(sampling_params) if sampling_params is not None else self.sampling_params
        semaphore = asyncio.Semaphore(self.max_concurrency) if self.max_concurrency else None
        started = time.perf_counter()
        total_tokens = 0

        async def run_one(index: int) -> tuple[int, VLLMGeneration]:
            if semaphore is None:
                return index, await self._generate_one(
                    prompt=prompts[index],
                    sampling_params=params,
                    context=contexts[index],
                    group_id=group_ids[index],
                    env_session_id=env_session_ids[index] or f"tau-{index}-{uuid.uuid4().hex[:8]}",
                )
            async with semaphore:
                return index, await self._generate_one(
                    prompt=prompts[index],
                    sampling_params=params,
                    context=contexts[index],
                    group_id=group_ids[index],
                    env_session_id=env_session_ids[index] or f"tau-{index}-{uuid.uuid4().hex[:8]}",
                )

        tasks = [asyncio.create_task(run_one(i)) for i in range(len(prompts))]
        try:
            for task in asyncio.as_completed(tasks):
                index, generation = await task
                total_tokens += generation.num_tokens
                elapsed = max(time.perf_counter() - started, 1e-9)
                self.last_metrics = {
                    "tokens": float(total_tokens),
                    "elapsed_s": elapsed,
                    "tps": total_tokens / elapsed,
                    "requests": float(len(prompts)),
                }
                yield index, generation
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = max(time.perf_counter() - started, 1e-9)
            self.last_metrics = {
                "tokens": float(total_tokens),
                "elapsed_s": elapsed,
                "tps": total_tokens / elapsed,
                "requests": float(len(prompts)),
            }
            if self.print_tps:
                print(
                    "[vLLM Rollout] "
                    f"requests={len(prompts)}, tokens={total_tokens}, "
                    f"elapsed={elapsed:.3f}s, TPS={total_tokens / elapsed:.2f}"
                )

    async def _generate_one(
        self,
        *,
        prompt: str,
        sampling_params: Any,
        context: Mapping[str, Any] | None,
        group_id: Any,
        env_session_id: str,
    ) -> VLLMGeneration:
        request_id = f"{self.request_id_prefix}-{uuid.uuid4().hex}"
        context_dict = dict(context or {})
        tool_context = dict(context_dict.get("tool_context") or context_dict)
        started = time.perf_counter()
        with rollout_context(env_session_id, tool_context):
            raw_output = await self._call_engine_generate(prompt, sampling_params, request_id)
            response, logprobs, token_ids = _extract_generation(raw_output)
            bound_context = get_current_tool_context()
            bound_env_session_id = get_current_env_session_id()
        merged_context: MutableMapping[str, Any] = dict(context_dict)
        merged_context.setdefault("env_session_id", bound_env_session_id)
        merged_context.setdefault("tool_context", bound_context)
        return VLLMGeneration(
            prompt=prompt,
            response=response,
            logprobs=logprobs,
            token_ids=tuple(token_ids or ()),
            request_id=request_id,
            group_id=group_id,
            context=dict(merged_context),
            env_session_id=bound_env_session_id,
            latency_s=time.perf_counter() - started,
        )

    async def _call_engine_generate(self, prompt: str, sampling_params: Any, request_id: str) -> Any:
        generated = self.engine.generate(prompt, sampling_params, request_id)
        if hasattr(generated, "__aiter__"):
            last = None
            async for output in generated:
                last = output
            return last
        if asyncio.iscoroutine(generated) or hasattr(generated, "__await__"):
            generated = await generated
        if hasattr(generated, "__aiter__"):
            last = None
            async for output in generated:
                last = output
            return last
        return generated

    @staticmethod
    def _build_engine(model: str | None, engine_args: Mapping[str, Any] | None) -> Any:
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
        except Exception as exc:  # pragma: no cover - exercised only without injection
            raise ImportError(
                "vLLM is not installed. Install vllm or pass engine=... to AsyncVLLMRollout."
            ) from exc
        kwargs = dict(engine_args or {})
        if model is not None:
            kwargs.setdefault("model", model)
        return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))

    @staticmethod
    def _build_sampling_params(params: Any | None) -> Any:
        if params is None:
            try:
                from vllm import SamplingParams

                return SamplingParams(temperature=1.0, max_tokens=256)
            except Exception:
                return {"temperature": 1.0, "max_tokens": 256}
        if isinstance(params, Mapping):
            try:
                from vllm import SamplingParams

                return SamplingParams(**dict(params))
            except Exception:
                return dict(params)
        return params


async def generate_and_score(
    rollout: AsyncVLLMRollout,
    prompts: Sequence[str],
    *,
    worker: Any | None = None,
    trainer: Any | None = None,
    sampling_params: Any | None = None,
    contexts: Sequence[Mapping[str, Any] | None] | None = None,
    group_ids: Sequence[Any] | None = None,
    env_session_ids: Sequence[str | None] | None = None,
    outcome_rewards: Sequence[float] | None = None,
    gather_fn: Any | None = None,
) -> RolloutBatch:
    """Sample with vLLM, stream-completed outputs to a scorer, then ingest.

    If ``worker`` is supplied, each completed response is immediately scored via
    ``worker.score_response`` to mimic online rollout-worker reward streaming.
    If ``trainer`` is supplied, the final aligned batch is passed to
    ``trainer.ingest_rollouts(...)`` and the returned samples are included in the
    result.
    """

    scorer = worker or getattr(trainer, "worker", None)
    rewards_by_index: dict[int, Any] = {}
    indexed: list[tuple[int, VLLMGeneration]] = []
    started = time.perf_counter()

    async for index, generation in rollout.generate_iter(
        prompts,
        sampling_params=sampling_params,
        contexts=contexts,
        group_ids=group_ids,
        env_session_ids=env_session_ids,
    ):
        indexed.append((index, generation))
        if scorer is not None and hasattr(scorer, "score_response"):
            rewards_by_index[index] = scorer.score_response(
                generation.prompt,
                generation.response,
                context=generation.context,
                outcome_reward=(outcome_rewards[index] if outcome_rewards is not None else 0.0),
            )

    indexed.sort(key=lambda item: item[0])
    generations = [generation for _, generation in indexed]
    prompts_out = [generation.prompt for generation in generations]
    responses = [generation.response for generation in generations]
    logprobs = [generation.logprobs for generation in generations]
    group_ids_out = [generation.group_id for generation in generations]
    contexts_out = [generation.context for generation in generations]
    total_tokens = sum(generation.num_tokens for generation in generations)
    elapsed = max(time.perf_counter() - started, 1e-9)
    samples: list[Any] = []

    if trainer is not None:
        samples = trainer.ingest_rollouts(
            prompts_out,
            responses,
            logprobs,
            group_ids=group_ids_out,
            contexts=contexts_out,
            outcome_rewards=outcome_rewards,
            gather_fn=gather_fn,
        )

    return RolloutBatch(
        prompts=prompts_out,
        responses=responses,
        logprobs=logprobs,
        group_ids=group_ids_out,
        contexts=contexts_out,
        generations=generations,
        rewards=[rewards_by_index[i] for i, _ in indexed if i in rewards_by_index],
        samples=list(samples),
        tps=total_tokens / elapsed,
        total_tokens=total_tokens,
        elapsed_s=elapsed,
    )


def _normalise_optional_sequence(values: Sequence[Any] | None, length: int, *, default: Any, name: str) -> list[Any]:
    if values is None:
        return [default for _ in range(length)]
    values = list(values)
    if len(values) != length:
        raise ValueError(f"{name} must have the same length as prompts")
    return values


def _extract_generation(output: Any) -> tuple[str, Any, Sequence[int]]:
    if output is None:
        return "", None, ()
    if isinstance(output, Mapping):
        if "outputs" in output:
            return _extract_generation_output(output["outputs"][0] if output["outputs"] else {})
        return _extract_generation_output(output)
    outputs = getattr(output, "outputs", None)
    if outputs is not None:
        return _extract_generation_output(outputs[0] if outputs else None)
    return _extract_generation_output(output)


def _extract_generation_output(candidate: Any) -> tuple[str, Any, Sequence[int]]:
    if candidate is None:
        return "", None, ()
    if isinstance(candidate, str):
        return candidate, None, ()
    if isinstance(candidate, Mapping):
        text = candidate.get("text") or candidate.get("response") or candidate.get("generated_text") or ""
        logprobs = candidate.get("logprobs", candidate.get("cumulative_logprob"))
        token_ids = candidate.get("token_ids") or candidate.get("output_token_ids") or ()
        return str(text), logprobs, tuple(token_ids or ())
    text = getattr(candidate, "text", None)
    if text is None:
        text = getattr(candidate, "response", None) or getattr(candidate, "generated_text", "")
    logprobs = getattr(candidate, "logprobs", None)
    if logprobs is None:
        logprobs = getattr(candidate, "cumulative_logprob", None)
    token_ids = getattr(candidate, "token_ids", None) or getattr(candidate, "output_token_ids", ())
    return str(text or ""), logprobs, tuple(token_ids or ())
