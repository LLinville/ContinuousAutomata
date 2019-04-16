
'''
Add velocity, add force restoring to zero
Phase/amplitude specific actions

Standing circular waves encouraged by averaging phase*amplitude and making it zero
'''

import numpy as np
import math

# def grad(values, ratio = 1.0):
#     return (ratio * np.roll(values, 1) - 2 * ratio * values + ratio * np.roll(values, -1))

def blur(values, ratio = 1.0):
    return (ratio * np.roll(values, 1) + values + ratio * np.roll(values, -1)) / (1 + 2 * ratio)

# def energy(positions, velocities):
#     return velocities * velocities +

class WaveAutomaton:
    def __init__(self, state_width=50):
        self.state_width = state_width
        self.phases = np.zeros(self.state_width)
        self.amplitudes = np.zeros(self.state_width)
        self.phase_velocities = np.zeros(self.state_width)
        self.amplitude_velocities = np.zeros(self.state_width)
        # self.phases = np.arange(0, 1, 1 / self.state_width)
        self.phases[10:12] = 0.1
        self.amplitudes[0 * self.state_width // 5 : 4 * self.state_width // 5] = 0.1
        self.phase_velocities[1 * self.state_width // 5 : 3 * self.state_width // 5] = 0.1

    def step(self, timestep=0.1):
        self.phase_velocities = blur(self.phase_velocities, 0.001)
        self.amplitudes += self.amplitude_velocities * timestep
        self.amplitude_velocities -= self.amplitudes * timestep
        self.phases += self.phase_velocities * timestep
        # self.phases = (np.roll(self.phases, 1) + self.phases + np.roll(self.phases, -1)) / 3
        # self.velocities = blur(self.velocities)
        # self.phases = blur(self.phases)
        self.phases %= 1 # 2 * np.pi


if __name__ == "__main__":
    pass
