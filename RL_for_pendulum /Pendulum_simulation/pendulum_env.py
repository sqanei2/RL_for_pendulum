import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .pendulum_dynamics import PendulumDynamics


class PendulumEnv(gym.Env):
    """
    Regular Pendulum Environment - starts at 60° and swings freely
    Goal: Agent should learn the natural pendulum dynamics (no control needed)
    """

    def __init__(self):
        super().__init__()
        self.dynamics = PendulumDynamics()

        # Action space: torque (should learn to apply zero torque)
        self.action_space = spaces.Box(
            low=-2.9, high=2.9, shape=(1,), dtype=np.float32  # Small range
        )

        # Observation space: [sin(θ), cos(θ), angular_velocity]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -10.0]),
            high=np.array([1.0, 1.0, 10.0]),
            dtype=np.float32
        )

        self.state = None
        self.steps = 0
        self.max_episode_steps = 850  # Shorter episodes

    def reset(self, seed=None, options=None):
        """Reset pendulum to 60° from bottom equilibrium"""
        super().reset(seed=seed)

        # Start at 60° from bottom (π/3 radians from bottom)
        initial_angle = np.pi-32*np.pi/180 # 179 from bottom equilibrium
        initial_velocity = 0.0  # Start from rest

        self.state = np.array([initial_angle, initial_velocity], dtype=np.float32)
        self.steps = 0

        observation = self._get_observation()
        info = {"theta": float(self.state[0]), "theta_dot": float(self.state[1])}
        return observation, info

    def step(self, action):
        """Execute one step - agent should learn to apply zero torque"""
        self.steps += 1

        # Extract torque (agent should learn this should be zero)
        torque = float(action[0]) if isinstance(action, np.ndarray) else float(action)

        # Update state using dynamics
        self.state = self.dynamics.update(self.state, torque)

        # Reward: encourage zero torque (learning natural dynamics)
        reward = self._calculate_reward(torque)  # Pass torque as argument

        # Episode ends after fixed time
        terminated = False
        truncated = self.steps >= self.max_episode_steps

        observation = self._get_observation()
        info = {
            "theta": float(self.state[0]),
            "theta_dot": float(self.state[1]),
            "torque": float(torque)
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Return [sin(θ), cos(θ), θ̇]"""
        theta, theta_dot = self.state
        return np.array([
            np.sin(theta),
            np.cos(theta),
            np.clip(theta_dot, -10.0, 10.0)
        ], dtype=np.float32)

    def _calculate_reward(self, torque):  # Fixed: now accepts torque parameter
        """Reward for learning natural dynamics (zero torque)"""
        theta, theta_dot = self.state

        # Reward for applying zero torque (learning natural behavior)
        torque_penalty = -100.0 * (torque ** 2)  # Heavy penalty for any torque

        # Small reward for realistic motion (not too fast)
        motion_reward = -0.1 * (theta_dot ** 2) if abs(theta_dot) > 5.0 else 0.1

        return torque_penalty + motion_reward
