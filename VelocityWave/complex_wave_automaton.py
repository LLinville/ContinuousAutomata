
'''
Add velocity, add force restoring to zero
Phase/amplitude specific actions
'''

import numpy as np
import math

# def grad(values, ratio = 1.0):
#     return (ratio * np.roll(values, 1) - 2 * ratio * values + ratio * np.roll(values, -1))

def blur(values, ratio = 1.0):
    return (ratio * np.roll(values, 1) + values + ratio * np.roll(values, -1)) / (1 + 2 * ratio)

def neighbor_averages(values):
    return np.roll(values, 1) + np.roll(values, -1) / 2

# def energy(positions, velocities):
#     return velocities * velocities +

class WaveAutomaton:
    def __init__(self, state_width=50):
        self.state_width = state_width
        # self.state = np.full(self.state_width, 0.1, dtype=np.cdouble)
        # self.state = np.zeros(self.state_width, dtype=np.cdouble)
        self.state = np.linspace(0, 0.1, state_width, dtype=np.cdouble)
        # self.state[0 * self.state_width // 5 : 1 * self.state_width // 5] = 0.01
        # self.state[2 * self.state_width // 5: 3 * self.state_width // 5] = 0.002
        self.velocity = np.zeros(self.state_width, dtype=np.cdouble)
        # self.velocity[0 * self.state_width // 5: 1 * self.state_width // 5] = 0.01j
        # self.velocity[2 * self.state_width // 5: 3 * self.state_width // 5] = 0.005j
        # self.velocity[2 * self.state_width // 5 : 3 * self.state_width // 5] = 0.1j
        # self.velocity += np.linspace(0, 0.1, state_width, dtype=np.cdouble) * 1j
        self.velocity += 1.1j
    def step(self, timestep=0.1):
        state_magnitude = self.state.real ** 2 + self.state.imag ** 2
        self.velocity -= self.state / np.maximum(np.sqrt(state_magnitude), 0.0001) * timestep / np.clip(state_magnitude, 0.001, 1)
        # self.velocity = blur(self.velocity, 0.005)
        # self.velocity -= (neighbor_averages(self.velocity) - self.velocity) * timestep
        self.state += self.velocity * timestep
        # self.state = blur(self.state, 0.0005)
        # self.phases = (np.roll(self.phases, 1) + self.phases + np.roll(self.phases, -1)) / 3
        # self.velocities = blur(self.velocities)
        # self.phases = blur(self.phases)
        # self.phases %= 1 # 2 * np.pi


if __name__ == "__main__":
    pass
