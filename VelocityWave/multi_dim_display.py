import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
from numpy import newaxis
from scipy.special import expit

from PIL import Image

from attractor_automaton import Automaton


class Display:
    def __init__(self):
        self.automaton = Automaton(state_width=300)
        self.state_history = [self.automaton.state]
        self.velocity_history = [self.automaton.velocity]

    def plot_states(self):
        state_history = np.array(self.state_history)
        state_history = state_history.transpose([1,0,2])
        # state_colors = np.stack((
        #     colors.Normalize()(np.sqrt(np.clip(np.abs(state_history[0]), -10000, 10000))),
        #     colors.Normalize()(np.sqrt(np.clip(np.abs(state_history[1]), -10000, 10000))),
        #     colors.Normalize()(np.sqrt(np.clip(np.abs(state_history[2]), -10000, 10000)))),
        #     axis=-1
        # )
        state_colors = np.stack((
            colors.Normalize()(np.clip(state_history[0], -10000, 10000)),
            colors.Normalize()(np.clip(state_history[1], -10000, 10000)),
            colors.Normalize()(np.clip(state_history[2], -10000, 10000))),
            axis=-1
        )
        # state_colors = colors.hsv_to_rgb(state_colors)
        img = Image.fromarray(np.asarray(np.rint(state_colors*255), dtype=np.uint8), 'RGB')
        img.save('states.png')
        # plt.imshow(state_colors, interpolation="None")
        # plt.savefig("phases.png", format='png')

    def plot_velocities(self):
        # phase_velocity_history = expit(np.array(self.phase_velocity_history)) * 2 - 1
        velocity_history = np.array(self.velocity_history)
        velocity_history = velocity_history.transpose([1, 0, 2])
        # state_colors = np.stack((
        #     (np.angle(velocity_history) + np.pi) / (2 * np.pi),
        #     np.ones_like(velocity_history).real,
        #     colors.Normalize()(np.sqrt(np.clip(np.abs(velocity_history), 0, 1000)))),
        #     axis=-1
        # )
        velocity_colors = np.stack((
            colors.Normalize()(np.sqrt(np.clip(np.abs(velocity_history[0]), -10000, 10000))),
            colors.Normalize()(np.sqrt(np.clip(np.abs(velocity_history[1]), -10000, 10000))),
            colors.Normalize()(np.sqrt(np.clip(np.abs(velocity_history[2]), -10000, 10000)))),
            axis=-1
        )
        # velocity_colors= colors.hsv_to_rgb(velocity_colors)
        img = Image.fromarray(np.asarray(np.rint(velocity_colors * 255), dtype=np.uint8), 'RGB')
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

    total_frames = 300
    steps_per_frame = 50
    for iteration in range(total_frames * steps_per_frame):
        display.automaton.step(timestep=0.005)
        if iteration % 100 == 0:
            print(iteration)
            print(
                np.average(np.linalg.norm(display.automaton.state, axis=0)),
                np.average(np.linalg.norm(display.automaton.velocity, axis=0)),
                np.average(np.linalg.norm(np.roll(display.automaton.state, 1) - display.automaton.state, axis=0)),
                np.average(np.linalg.norm(display.automaton.pull, axis=0))
            )
        if iteration % steps_per_frame == 0:

            display.state_history.append(np.copy(display.automaton.state))
            display.velocity_history.append(np.copy(display.automaton.velocity))
    display.plot_velocities()
    display.plot_states()
    print("dummyline")
