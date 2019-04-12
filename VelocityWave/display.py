import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from numpy import newaxis
from scipy.special import expit

from wave_automaton import WaveAutomaton


class Display:
    def __init__(self):
        self.automaton = WaveAutomaton(state_width=100)
        self.phase_state_history = [self.automaton.phases]
        self.velocity_state_history = [self.automaton.velocities]

    def plot_state(self):
        state_colors = np.tile(np.array(self.phase_state_history)[:,:,newaxis], 3)

        max = state_colors.max()
        min = state_colors.min()
        state_colors = expit(state_colors / (max - min))
        # if abs(max) > 0:
        #     state_colors -= min
        #     state_colors /= max - min

        plt.imshow(state_colors, interpolation="None")
        plt.savefig("phases.png", format='png')

        state_colors = np.tile(np.array(self.velocity_state_history)[:, :, newaxis], 3)
        max = state_colors.max()
        min = state_colors.min()
        if max - min > 0.00001:
            state_colors = expit(state_colors / (max - min))

        # if abs(max) > 0:
        #     state_colors += min
        #     state_colors /= max - min

        plt.imshow(state_colors, interpolation="None")
        plt.savefig("velocities.png", format='png')

if __name__ == "__main__":
    display = Display()

    steps_per_frame = 10
    for iteration in range(200 * steps_per_frame):
        display.automaton.step(timestep=0.1)
        if iteration % steps_per_frame == 0:
            display.phase_state_history.append(np.copy(display.automaton.phases))
            display.velocity_state_history.append(np.copy(display.automaton.velocities))
    display.plot_state()
    print("dummyline")
