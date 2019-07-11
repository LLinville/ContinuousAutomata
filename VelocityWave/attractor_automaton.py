
import numpy as np
import math
import collections

Points = collections.namedtuple('Points', 'x y z')

# def grad(values, ratio = 1.0):
#     return (ratio * np.roll(values, 1) - 2 * ratio * values + ratio * np.roll(values, -1))

def blur(values, ratio = 1.0):
    return (ratio * np.roll(values, 1) + values + ratio * np.roll(values, -1)) / (1 + 2 * ratio)

def neighbor_averages(values):
    return np.roll(values, 1) + np.roll(values, -1) / 2

def attractor(values):
    a,b,c = 1,1,1

    #lorenz
    x, y, z = values.x, values.y, values.z
    return Points(a*(y-x), x*(b-z)-y, x*y-c*z)

# def energy(positions, velocities):
#     return velocities * velocities +

class Automaton:
    def __init__(self, state_width=50):
        self.state = Points(
            np.linspace(0, 2, state_width),
            np.linspace(0, 2, state_width),
            np.linspace(0, 2, state_width)
        )
        self.velocity = Points(np.zeros(state_width), np.zeros(state_width), np.zeros(state_width))

    def step(self, timestep=1):
        self.velocity = attractor(self.state)
        # self.state = blur(self.state) + timestep * self.velocity


if __name__ == "__main__":
    pass
