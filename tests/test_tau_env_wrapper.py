import asyncio

from envs.tau_env_wrapper import TauAirlineEnvWrapper


class FakeAirlineEnv:
    def __init__(self):
        self.task_id = None
        self.local_step = 0

    def reset(self, task_id=None):
        self.task_id = task_id
        self.local_step = 0
        return {
            "observation": {"task_id": task_id, "message": "initial"},
            "info": {
                "episode_id": f"episode-{task_id}",
                "process_metrics": {"tool_calls": 0},
            },
        }

    def step(self, action):
        self.local_step += 1
        reward = 1.0 if action == f"finish-{self.task_id}" else 0.25
        done = self.local_step >= 2
        return {
            "observation": {
                "task_id": self.task_id,
                "action": action,
                "local_step": self.local_step,
            },
            "reward": reward,
            "done": done,
            "info": {
                "process_metrics": {
                    "tool_calls": self.local_step,
                    "success": done and reward == 1.0,
                }
            },
        }


def test_reset_and_step_return_verl_sampler_shape():
    wrapper = TauAirlineEnvWrapper(env_factory=FakeAirlineEnv)

    reset_output = wrapper.reset(task_id="A")

    assert set(reset_output) == {"observation", "reward", "done", "info"}
    assert reset_output["observation"] == {"task_id": "A", "message": "initial"}
    assert reset_output["reward"] == 0.0
    assert reset_output["done"] is False
    assert reset_output["info"]["episode_id"] == "episode-A"
    assert reset_output["info"]["step_count"] == 0
    assert reset_output["info"]["process_metrics"]["tool_calls"] == 0

    step_output = wrapper.step("search-flight")

    assert set(step_output) == {"observation", "reward", "done", "info"}
    assert step_output["observation"]["task_id"] == "A"
    assert step_output["observation"]["local_step"] == 1
    assert step_output["reward"] == 0.25
    assert step_output["done"] is False
    assert step_output["info"]["step_count"] == 1
    assert step_output["info"]["total_reward"] == 0.25
    assert step_output["info"]["process_metrics"]["tool_calls"] == 1


async def _run_rollout(wrapper, task_id, first_action):
    reset_output = wrapper.reset(task_id=task_id)
    state_after_reset = wrapper.get_current_env_state()
    await asyncio.sleep(0)
    first_step = wrapper.step(first_action)
    await asyncio.sleep(0)
    second_step = wrapper.step(f"finish-{task_id}")
    final_state = wrapper.get_current_env_state()
    return reset_output, first_step, second_step, state_after_reset, final_state


def test_contextvars_isolate_concurrent_async_rollout_state():
    wrapper = TauAirlineEnvWrapper(env_factory=FakeAirlineEnv)

    async def main():
        return await asyncio.gather(
            _run_rollout(wrapper, "A", "inspect-A"),
            _run_rollout(wrapper, "B", "inspect-B"),
        )

    rollout_a, rollout_b = asyncio.run(main())

    assert rollout_a[0]["observation"]["task_id"] == "A"
    assert rollout_b[0]["observation"]["task_id"] == "B"
    assert rollout_a[1]["observation"]["task_id"] == "A"
    assert rollout_b[1]["observation"]["task_id"] == "B"
    assert rollout_a[2]["done"] is True
    assert rollout_b[2]["done"] is True

    state_a_after_reset, state_a_final = rollout_a[3], rollout_a[4]
    state_b_after_reset, state_b_final = rollout_b[3], rollout_b[4]

    assert state_a_after_reset is state_a_final
    assert state_b_after_reset is state_b_final
    assert state_a_final is not state_b_final
    assert state_a_final.task_id == "A"
    assert state_b_final.task_id == "B"
    assert state_a_final.step_count == 2
    assert state_b_final.step_count == 2
    assert state_a_final.last_observation["task_id"] == "A"
    assert state_b_final.last_observation["task_id"] == "B"
