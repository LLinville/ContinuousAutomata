import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
from numpy import newaxis
from scipy.special import expit

from PIL import Image

from complex_wave_automaton import WaveAutomaton


class Display:
    def __init__(self):
        self.automaton = WaveAutomaton(state_width=100)
        self.state_history = [self.automaton.state]
        self.velocity_history = [self.automaton.velocity]

    def plot_states(self):
        state_history = self.state_history
        state_colors = np.stack((
            (np.angle(state_history) + np.pi) / (2*np.pi),
            np.ones_like(state_history).real,
            colors.Normalize()(np.sqrt(np.clip(np.abs(state_history), 0, 1)))),
            axis=-1
        )
        state_colors = colors.hsv_to_rgb(state_colors)
        img = Image.fromarray(np.asarray(np.rint(state_colors*255), dtype=np.uint8), 'RGB')
        img.save('states.png')
        # plt.imshow(state_colors, interpolation="None")
        # plt.savefig("phases.png", format='png')

    def plot_velocities(self):
        # phase_velocity_history = expit(np.array(self.phase_velocity_history)) * 2 - 1
        velocity_history = self.velocity_history
        state_colors = np.stack((
            (np.angle(velocity_history) + np.pi) / (2 * np.pi),
            np.ones_like(velocity_history).real,
            colors.Normalize()(np.sqrt(np.clip(np.abs(velocity_history), 0, 1000)))),
            axis=-1
        )
        state_colors = colors.hsv_to_rgb(state_colors)
        img = Image.fromarray(np.asarray(np.rint(state_colors * 255), dtype=np.uint8), 'RGB')
        img.save('phase_velocities.png')
        # plt.imshow(state_colors, interpolation="None")
        # plt.savefig("phase_velocities.png", format='png')

    # def plot_state(self):
    #     state_colors = np.tile(np.array(self.phase_state_history)[:,:,newaxis], 3)
    #
    #     max = state_colors.max()
    #     min = state_colors.min()
    #     if abs(max - min) > 0.00001:
    #         state_colors = expit(state_colors / (max - min))
    #     # if abs(max) > 0:
    #     #     state_colors -= min
    #     #     state_colors /= max - min
    #
    #     plt.imshow(state_colors, interpolation="None")
    #     plt.savefig("phases.png", format='png')
    #
    #     state_colors = np.tile(np.array(self.phase_velocity_history)[:, :, newaxis], 3)
    #     max = state_colors.max()
    #     min = state_colors.min()
    #     if max - min > 0.00001:
    #         state_colors = expit(state_colors / (max - min))
    #
    #     # if abs(max) > 0:
    #     #     state_colors += min
    #     #     state_colors /= max - min
    #
    #     plt.imshow(state_colors, interpolation="None")
    #     plt.savefig("velocities.png", format='png')

if __name__ == "__main__":
    display = Display()

    steps_per_frame = 10
    for iteration in range(2000 * steps_per_frame):
        if iteration % 10 == 0:
            print(iteration)

        display.automaton.step(timestep=0.001)
        if iteration % steps_per_frame == 0:
            display.state_history.append(np.copy(display.automaton.state))
            display.velocity_history.append(np.copy(display.automaton.velocity))
    # display.plot_state()
    display.plot_velocities()
    display.plot_states()
    print("dummyline")
