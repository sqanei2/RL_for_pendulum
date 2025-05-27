import gymnasium as gym  # Changed from gym
from gymnasium.envs.registration import register  # Updated import
from .pendulum_env import PendulumEnv

register(
    id='CustomPendulum-v0',
    entry_point='Pendulum_simulation.pendulum_env:PendulumEnv',
    max_episode_steps=500,
)
