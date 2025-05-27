import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from Pendulum_simulation.register_envs import *
import os


class CheckpointCallback(BaseCallback):
    """Save model at specific training steps"""

    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.checkpoints = [0.25, 0.5, 0.75, 1.0]  # 25%, 50%, 75%, 100%

    def _on_step(self) -> bool:
        # Calculate training progress
        progress = self.num_timesteps / self.model.learning_starts

        for checkpoint in self.checkpoints:
            if abs(progress - checkpoint) < 0.01:  # Within 1% of checkpoint
                checkpoint_path = f"{self.save_path}_checkpoint_{int(checkpoint * 100)}pct"
                self.model.save(checkpoint_path)
                print(f"Saved checkpoint at {int(checkpoint * 100)}% training")
                self.checkpoints.remove(checkpoint)  # Don't save again
                break
        return True


def train_with_checkpoints():
    print("Training agent with intermediate checkpoints...")

    # Create environment
    env = gym.make('CustomPendulum-v0')

    # Create PPO agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="regular_pendulum_ppo"
    )

    # Train with checkpoints
    total_steps = 50000
    steps_per_checkpoint = total_steps // 4

    # Save untrained model
    model.save("regular_pendulum_ppo_checkpoint_0pct")
    print("Saved untrained model (0%)")

    # Train in stages and save checkpoints
    for i, pct in enumerate([25, 50, 75, 100]):
        print(f"\nTraining to {pct}%...")
        model.learn(total_timesteps=steps_per_checkpoint)
        model.save(f"regular_pendulum_ppo_checkpoint_{pct}pct")
        print(f"Saved checkpoint at {pct}% training")

    print("Training completed with all checkpoints saved!")


if __name__ == "__main__":
    train_with_checkpoints()
