import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
from numpy import newaxis
from scipy.special import expit

from polar_wave_automaton import WaveAutomaton


class Display:
    def __init__(self):
        self.automaton = WaveAutomaton(state_width=100)
        self.phase_state_history = [self.automaton.phases]
        self.phase_velocity_history = [self.automaton.phase_velocities]
        self.amplitude_state_history = [self.automaton.amplitudes]
        self.amplitude_velocity_history = [self.automaton.amplitude_velocities]

    def plot_phases(self):
        phase_state_history = np.array(self.phase_state_history)
        state_colors = np.stack((
            phase_state_history,
            np.ones_like(phase_state_history),
            np.ones_like(phase_state_history)),
            axis=-1
        )
        state_colors = colors.hsv_to_rgb(state_colors)
        plt.imshow(state_colors, interpolation="None")
        plt.savefig("phases.png", format='png')

    def plot_phase_velocities(self):
        # phase_velocity_history = expit(np.array(self.phase_velocity_history)) * 2 - 1
        phase_velocity_history = colors.Normalize()(self.phase_velocity_history)
        state_colors = np.tile(np.array(phase_velocity_history)[:, :, newaxis], 3)
        plt.imshow(state_colors, interpolation="None")
        plt.savefig("phase_velocities.png", format='png')

    def plot_state(self):
        state_colors = np.tile(np.array(self.phase_state_history)[:,:,newaxis], 3)

        max = state_colors.max()
        min = state_colors.min()
        if abs(max - min) > 0.00001:
            state_colors = expit(state_colors / (max - min))
        # if abs(max) > 0:
        #     state_colors -= min
        #     state_colors /= max - min

        plt.imshow(state_colors, interpolation="None")
        plt.savefig("phases.png", format='png')

        state_colors = np.tile(np.array(self.phase_velocity_history)[:, :, newaxis], 3)
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

    steps_per_frame = 100
    for iteration in range(200 * steps_per_frame):
        display.automaton.step(timestep=0.001)
        if iteration % steps_per_frame == 0:
            display.phase_state_history.append(np.copy(display.automaton.phases))
            display.phase_velocity_history.append(np.copy(display.automaton.phase_velocities))
    # display.plot_state()
    display.plot_phase_velocities()
    display.plot_phases()
    print("dummyline")
