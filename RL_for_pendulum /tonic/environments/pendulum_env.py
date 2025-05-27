import numpy as np

class PendulumDynamics:
    def __init__(self):
        self.m = 0.1
        self.l = 1
        self.R = 0.09
        self.k = 2
        self.c = 0.1
        self.g = 9.81
        self.F = 1

        self.A = np.array([[0, 1], [self.g * self.l, -self.c/self.m]])
        self.B = np.array([[0], [self.F/(self.m * self.l)]])

    def step(self, x, u, dt=0.02):
        dx = self.A @ x.reshape(-1,1) + self.B * u
        x_new = x.reshape(-1,1) + dx * dt
        return x_new.flatten()
