"""veRL-compatible adapter for tau-bench AirlineEnv.

The wrapper deliberately keeps per-rollout mutable environment state inside a
``contextvars.ContextVar``.  A single ``TauAirlineEnvWrapper`` instance can
therefore be shared by many asyncio tasks or threaded rollout workers without
one rollout overwriting another rollout's active ``AirlineEnv`` instance,
step counter, or process metrics.
"""

from __future__ import annotations

import contextvars
import importlib
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple

EnvFactory = Callable[..., Any]


_current_env_state: contextvars.ContextVar["RolloutEnvState | None"] = contextvars.ContextVar(
    "tau_airline_env_state", default=None
)


@dataclass
class RolloutEnvState:
    """Mutable state scoped to one rollout context."""

    env: Any
    episode_id: str
    step_count: int = 0
    done: bool = False
    total_reward: float = 0.0
    started_at: float = field(default_factory=time.time)
    last_observation: Any = None
    task_id: Optional[str] = None
    process_metrics: Dict[str, Any] = field(default_factory=dict)


class TauAirlineEnvWrapper:
    """Context-isolated adapter around tau-bench's ``AirlineEnv``.

    Parameters
    ----------
    env_factory:
        Optional factory used to construct an AirlineEnv.  If omitted, the
        wrapper lazily imports tau-bench and resolves an ``AirlineEnv`` class
        from known module paths.
    env_kwargs:
        Keyword arguments forwarded to the environment constructor.
    max_steps:
        Optional hard episode horizon enforced by the wrapper.

    Both ``reset`` and ``step`` return dictionaries with the veRL sampler shape:
    ``{"observation": ..., "reward": float, "done": bool, "info": dict}``.
    """

    def __init__(
        self,
        env_factory: Optional[EnvFactory] = None,
        env_kwargs: Optional[Mapping[str, Any]] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        self.env_factory = env_factory or self._load_airline_env_factory()
        self.env_kwargs = dict(env_kwargs or {})
        self.max_steps = max_steps
        self._episode_seq = 0

    def reset(self, *, task_id: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Start a new rollout in the current context and return initial data."""

        env = self.env_factory(**self.env_kwargs)
        reset_result = self._call_with_supported_kwargs(env.reset, task_id=task_id, **kwargs)
        observation, info = self._normalize_reset_result(reset_result)

        self._episode_seq += 1
        episode_id = str(info.get("episode_id") or task_id or f"episode-{self._episode_seq}")
        state = RolloutEnvState(
            env=env,
            episode_id=episode_id,
            last_observation=observation,
            task_id=task_id,
            process_metrics=self._extract_process_metrics(info),
        )
        _current_env_state.set(state)

        info = self._build_info(state, info)
        return self._verl_transition(observation=observation, reward=0.0, done=False, info=info)

    def step(self, action: Any) -> Dict[str, Any]:
        """Advance the current context's AirlineEnv by one action."""

        state = _current_env_state.get()
        if state is None:
            raise RuntimeError("TauAirlineEnvWrapper.step() called before reset() in this context")
        if state.done:
            return self._verl_transition(
                observation=state.last_observation,
                reward=0.0,
                done=True,
                info=self._build_info(state, {"already_done": True}),
            )

        step_result = state.env.step(action)
        observation, reward, done, info = self._normalize_step_result(step_result)

        state.step_count += 1
        state.total_reward += float(reward)
        if self.max_steps is not None and state.step_count >= self.max_steps:
            done = True
            info.setdefault("truncated", True)
            info.setdefault("termination_reason", "max_steps")
        state.done = bool(done)
        state.last_observation = observation
        state.process_metrics.update(self._extract_process_metrics(info))
        _current_env_state.set(state)

        return self._verl_transition(
            observation=observation,
            reward=float(reward),
            done=state.done,
            info=self._build_info(state, info),
        )

    @staticmethod
    def get_current_env_state() -> Optional[RolloutEnvState]:
        """Return the rollout state bound to the current context, if any."""

        return _current_env_state.get()

    @classmethod
    def clear_current_env_state(cls) -> None:
        """Clear rollout state for the current context."""

        _current_env_state.set(None)

    @staticmethod
    def _load_airline_env_factory() -> EnvFactory:
        candidates = (
            "tau_bench.envs.airline.env",
            "tau_bench.envs.airline",
            "tau_bench.envs.airline.environment",
        )
        for module_name in candidates:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            airline_env = getattr(module, "AirlineEnv", None)
            if airline_env is not None:
                return airline_env
        raise ImportError(
            "Could not import tau-bench AirlineEnv. Install tau-bench or pass "
            "env_factory=... to TauAirlineEnvWrapper."
        )

    @staticmethod
    def _call_with_supported_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return func(**clean_kwargs)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return func(**clean_kwargs)
        supported = {k: v for k, v in clean_kwargs.items() if k in signature.parameters}
        return func(**supported)

    @staticmethod
    def _normalize_reset_result(result: Any) -> Tuple[Any, Dict[str, Any]]:
        if isinstance(result, Mapping):
            if "observation" in result:
                info = dict(result.get("info") or {})
                for key, value in result.items():
                    if key not in {"observation", "info"}:
                        info.setdefault(key, value)
                return result["observation"], info
            return dict(result), {}
        if isinstance(result, tuple):
            if len(result) == 2:
                observation, info = result
                return observation, dict(info or {})
            if len(result) >= 1:
                return result[0], {}
        return result, {}

    @staticmethod
    def _normalize_step_result(result: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        if isinstance(result, Mapping):
            observation = result.get("observation", result.get("obs"))
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", result.get("terminated", False) or result.get("truncated", False)))
            info = dict(result.get("info") or {})
            for key, value in result.items():
                if key not in {"observation", "obs", "reward", "done", "terminated", "truncated", "info"}:
                    info.setdefault(key, value)
            if "terminated" in result:
                info.setdefault("terminated", result["terminated"])
            if "truncated" in result:
                info.setdefault("truncated", result["truncated"])
            return observation, reward, done, info

        if isinstance(result, tuple):
            if len(result) == 5:
                observation, reward, terminated, truncated, info = result
                info = dict(info or {})
                info.setdefault("terminated", terminated)
                info.setdefault("truncated", truncated)
                return observation, float(reward), bool(terminated or truncated), info
            if len(result) == 4:
                observation, reward, done, info = result
                return observation, float(reward), bool(done), dict(info or {})
        raise TypeError(f"Unsupported AirlineEnv.step() result format: {type(result)!r}")

    @staticmethod
    def _extract_process_metrics(info: Mapping[str, Any]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        process_metrics = info.get("process_metrics")
        if isinstance(process_metrics, Mapping):
            metrics.update(process_metrics)
        for key in (
            "tool_calls",
            "api_calls",
            "invalid_actions",
            "success",
            "pass^1",
            "terminated",
            "truncated",
            "termination_reason",
        ):
            if key in info:
                metrics[key] = info[key]
        return metrics

    @staticmethod
    def _build_info(state: RolloutEnvState, env_info: Mapping[str, Any]) -> Dict[str, Any]:
        info = dict(env_info)
        info.setdefault("episode_id", state.episode_id)
        info.setdefault("task_id", state.task_id)
        info["step_count"] = state.step_count
        info["total_reward"] = state.total_reward
        info["elapsed_seconds"] = time.time() - state.started_at
        info["process_metrics"] = dict(state.process_metrics)
        return info

    @staticmethod
    def _verl_transition(observation: Any, reward: float, done: bool, info: MutableMapping[str, Any]) -> Dict[str, Any]:
        return {
            "observation": observation,
            "reward": float(reward),
            "done": bool(done),
            "info": dict(info),
        }


__all__ = ["RolloutEnvState", "TauAirlineEnvWrapper"]
