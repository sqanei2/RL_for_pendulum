import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from Pendulum_simulation.register_envs import *
from Pendulum_simulation.pendulum_dynamics import PendulumDynamics


def test_angle(angle_deg):
    """Test agent vs true physics from specific angle"""
    print(f"Testing from {angle_deg}°...")

    # Load trained model
    model = PPO.load("regular_pendulum_ppo_checkpoint_100pct")

    # Test agent
    env = gym.make('CustomPendulum-v0')
    obs, _ = env.reset(seed=42)

    # Set angle
    angle_rad = np.radians(angle_deg)
    env.unwrapped.state = np.array([angle_rad, 0.0])
    obs = env.unwrapped._get_observation()

    agent_angles = []
    agent_actions = []

    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        agent_angles.append(np.degrees(info['theta']))
        agent_actions.append(action[0])
        if terminated or truncated:
            break

    # True physics
    dynamics = PendulumDynamics()
    state = np.array([angle_rad, 0.0])
    true_angles = []

    for step in range(len(agent_angles)):
        state = dynamics.update(state, torque=0.0)
        true_angles.append(np.degrees(state[0]))

    # Plot
    times = [i * 0.02 for i in range(len(agent_angles))]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(times, true_angles, 'k-', linewidth=3, label='True Physics')
    plt.plot(times, agent_angles, 'b-', linewidth=2, label='Agent')
    plt.axhline(y=angle_deg, color='r', linestyle='--', alpha=0.5, label=f'Start: {angle_deg}°')
    plt.title(f'Response from {angle_deg}°')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(times, agent_actions, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Zero Torque')
    plt.title('Agent Actions')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (N⋅m)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Results
    angle_error = np.mean(np.abs(np.array(true_angles) - np.array(agent_angles)))
    avg_action = np.mean(np.abs(agent_actions))

    print(f"Angle Error: {angle_error:.1f}°")
    print(f"Avg Action: {avg_action:.4f}")
    print(f"Agent Final: {agent_angles[-1]:.1f}°")
    print(f"Physics Final: {true_angles[-1]:.1f}°")


if __name__ == "__main__":
    import sys

    angle = int(sys.argv[1]) if len(sys.argv) > 1 else 179
    test_angle(angle)
