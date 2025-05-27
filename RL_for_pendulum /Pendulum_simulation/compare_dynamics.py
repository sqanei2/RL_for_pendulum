import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from Pendulum_simulation.register_envs import *
from Pendulum_simulation.pendulum_dynamics import PendulumDynamics
#gfgfgffhg

def test_multiple_agents(angle_deg):
    """Test multiple agent training stages vs true physics from specific angle"""
    print(f"Testing multiple agents from {angle_deg}°...")

    # Agent configurations
    agents = [
        ("untrained", "Untrained", 'red', '--'),
        ("regular_pendulum_ppo_checkpoint_50pct", "Half Trained (50%)", 'orange', '-.'),
        ("regular_pendulum_ppo_checkpoint_100pct", "Fully Trained (100%)", 'blue', '-')
    ]

    # Set angle
    angle_rad = np.radians(angle_deg)

    # True physics simulation
    dynamics = PendulumDynamics()
    state = np.array([angle_rad, 0.0])
    true_angles = []

    # Simulate for 17 seconds
    for step in range(850):
        state = dynamics.update(state, torque=0.0)
        true_angles.append(np.degrees(state[0]))

    true_times = [i * 0.02 for i in range(len(true_angles))]

    # Test all agents
    agent_data = {}

    for model_path, label, color, style in agents:
        print(f"Testing {label}...")

        # Create/load model
        env = gym.make('CustomPendulum-v0')

        if model_path == "untrained":
            model = PPO("MlpPolicy", env, verbose=0)  # Fresh untrained model
        else:
            try:
                model = PPO.load(model_path)
            except:
                print(f"Could not load {model_path}, using untrained model")
                model = PPO("MlpPolicy", env, verbose=0)

        # Reset and set angle
        obs, _ = env.reset(seed=42)
        env.unwrapped.state = np.array([angle_rad, 0.0])
        obs = env.unwrapped._get_observation()

        agent_angles = []
        agent_actions = []

        # Simulate for 17 seconds
        for step in range(850):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            agent_angles.append(np.degrees(info['theta']))
            agent_actions.append(action[0])
            if terminated or truncated:
                break

        # Store data
        agent_data[label] = {
            'times': [i * 0.02 for i in range(len(agent_angles))],
            'angles': agent_angles,
            'actions': agent_actions,
            'color': color,
            'style': style
        }

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Angle comparison
    axes[0, 0].plot(true_times, true_angles, 'k-', linewidth=3, label='True Physics')
    for label, data in agent_data.items():
        axes[0, 0].plot(data['times'], data['angles'],
                        color=data['color'], linestyle=data['style'],
                        linewidth=2, label=label)
    axes[0, 0].axhline(y=angle_deg, color='gray', linestyle=':', alpha=0.5, label=f'Start: {angle_deg}°')
    axes[0, 0].set_title(f'Angle Response from {angle_deg}° (17 seconds)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Angle (degrees)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Actions comparison
    for label, data in agent_data.items():
        axes[0, 1].plot(data['times'], data['actions'],
                        color=data['color'], linestyle=data['style'],
                        linewidth=2, label=label)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Zero Torque (Physics)')
    axes[0, 1].set_title('Agent Actions Comparison')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Torque (N⋅m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Phase plot
    axes[1, 0].plot(true_angles, [0] * len(true_angles), 'k-', linewidth=3, label='True Physics')
    for label, data in agent_data.items():
        # Calculate velocities (approximate derivative)
        velocities = np.gradient(data['angles']) / 0.02
        axes[1, 0].plot(data['angles'], velocities,
                        color=data['color'], linestyle=data['style'],
                        linewidth=2, label=label)
    axes[1, 0].set_title('Phase Plot (Angle vs Velocity)')
    axes[1, 0].set_xlabel('Angle (degrees)')
    axes[1, 0].set_ylabel('Angular Velocity (deg/s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Performance metrics
    agent_names = list(agent_data.keys())
    angle_errors = []
    avg_actions = []

    for label, data in agent_data.items():
        # Calculate angle error compared to true physics
        min_len = min(len(true_angles), len(data['angles']))
        error = np.mean(np.abs(np.array(true_angles[:min_len]) - np.array(data['angles'][:min_len])))
        angle_errors.append(error)

        # Calculate average action magnitude
        avg_action = np.mean(np.abs(data['actions']))
        avg_actions.append(avg_action)

    x_pos = np.arange(len(agent_names))
    width = 0.35

    bars1 = axes[1, 1].bar(x_pos - width / 2, angle_errors, width, label='Angle Error (°)', color='lightcoral')
    ax_twin = axes[1, 1].twinx()
    bars2 = ax_twin.bar(x_pos + width / 2, avg_actions, width, label='Avg Torque', color='lightblue')

    axes[1, 1].set_xlabel('Agent Type')
    axes[1, 1].set_ylabel('Angle Error (degrees)', color='red')
    ax_twin.set_ylabel('Average Torque Magnitude', color='blue')
    axes[1, 1].set_title('Performance Comparison')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(agent_names, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars1, angle_errors):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 1,
                        f'{value:.1f}°', ha='center', va='bottom', fontsize=9)

    for bar, value in zip(bars2, avg_actions):
        height = bar.get_height()
        ax_twin.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.suptitle(f'Agent Training Progression: {angle_deg}° Start Position', fontsize=16, y=0.98)
    plt.show()

    # Print results
    print(f"\n{'=' * 60}")
    print(f"PERFORMANCE ANALYSIS FROM {angle_deg}°")
    print(f"{'=' * 60}")

    for i, (label, data) in enumerate(agent_data.items()):
        print(f"\n{label}:")
        print(f"  Angle Error: {angle_errors[i]:.1f}°")
        print(f"  Avg Action: {avg_actions[i]:.4f}")
        print(f"  Final Angle: {data['angles'][-1]:.1f}°")

        if angle_errors[i] < 10 and avg_actions[i] < 0.1:
            status = "✅ EXCELLENT"
        elif angle_errors[i] < 30:
            status = "✅ GOOD"
        else:
            status = "⚠️  POOR"
        print(f"  Status: {status}")


if __name__ == "__main__":
    import sys

    angle = int(sys.argv[1]) if len(sys.argv) > 1 else 179
    test_multiple_agents(angle)
