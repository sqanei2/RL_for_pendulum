import gym
from .pendulum_env import PendulumEnv

class TonicPendulumWrapper(gym.Wrapper):
    def __init__(self):
        env = PendulumEnv()
        super().__init__(env)

    def initialize(self, seed=None):
        """Initialize the environment with a seed for Tonic compatibility"""
        if seed is not None:
            # Replace seed() with reset() for modern Gym API
            self.env.reset(seed=seed)
        return self
