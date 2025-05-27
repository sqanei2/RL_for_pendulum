import numpy as np


class PendulumDynamics:
    def __init__(self, mass=1.0, length=1.0, gravity=9.81, damping=0.0, dt=0.02):
        self.mass = mass  # Mass of pendulum bob (kg)
        self.length = length  # Length of pendulum (m)
        self.gravity = gravity  # Gravitational acceleration (m/s²)
        self.damping = damping  # Damping coefficient
        self.dt = dt  # Time step (s)

    def update(self, state, torque=0.0):
        """Update pendulum state - regular pendulum hanging down"""
        theta, theta_dot = state

        # Regular pendulum equation: θ̈ = -(g/L)sin(θ) - bθ̇ + τ/(mL²)
        # For free motion: τ = 0 (no external torque)
        moment_of_inertia = self.mass * self.length ** 2

        theta_ddot = (
                -(self.gravity / self.length) * np.sin(theta) -  # Gravity restoring force
                self.damping * theta_dot +  # Damping
                torque / moment_of_inertia  # External torque (should be 0)
        )

        # Integrate using Euler method
        new_theta_dot = theta_dot + theta_ddot * self.dt
        new_theta = theta + new_theta_dot * self.dt

        return np.array([new_theta, new_theta_dot])
