from matplotlib import pyplot as plt
import numpy as np
from numpy import newaxis
from wave_automaton import WaveAutomaton


class Display:
    def __init__(self):
        self.automaton = WaveAutomaton()
        self.state_history = [self.automaton.velocities]

    def plot_state(self):
        state_colors = np.tile(np.array(self.state_history)[:,:,newaxis], 3)
        plt.imshow(state_colors)


if __name__ == "__main__":
    display = Display()

    for iteration in range(5):
        display.automaton.step()
        display.state_history.append(display.automaton.velocities)
    display.plot_state()
    plt.show()

