
'''
Add velocity, add force restoring to zero
Phase/amplitude specific actions
'''

import numpy as np
import math




class WaveAutomaton:
    def __init__(self):
        self.state_width = 100
        self.phases = np.zeros(self.state_width)
        self.amplitudes = np.zeros(self.state_width)
        self.velocities = np.arange(0, 2*math.pi, 2*math.pi / self.state_width) # Speed in phase space in units of 2pi

    def step(self, timestep = 1):
        self.phases += self.velocities * timestep


if __name__ == "__main__":
    pass
