import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import gym_robosuite  # noqa: F401


@pytest.mark.parametrize(
    "env_task, obs_type",
    [
        ("Square_D0-v0", "pixels"),
    ],
)
def test_aloha(env_task, obs_type):
    env = gym.make(f"gym_robosuite/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped)
