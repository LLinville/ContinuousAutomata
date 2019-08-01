
import numpy as np
import math
import collections

from simplexnoise import scaled_raw_noise_2d

# Points = collections.namedtuple('Points', 'x y z')

# def grad(values, ratio = 1.0):
#     return (ratio * np.roll(values, 1) - 2 * ratio * values + ratio * np.roll(values, -1))

def blur(values, ratio = 1.0):
    return (ratio * np.roll(values, 1, axis=1) + values + ratio * np.roll(values, -1, axis=1)) / (1 + 2 * ratio)

def neighbor_averages(values, width=5):
    # return (np.roll(values, 1) + np.roll(values, -1)) / 2
    total = np.zeros_like(values)
    for i in range(1, width):
        total += np.roll(values, -i, axis=1)/i + np.roll(values, i, axis=1)/i
    return total / (width * 2)
    # return (np.roll(values, 2) + np.roll(values, 1) + np.roll(values, -1) + np.roll(values, -2))/4

def lorentz_attractor(values):
    a,b,c = 10, 28, 8.0/3

    #lorenz
    x, y, z = values[0], values[1], values[2]
    return np.stack((
        a*(y-x),
        x*(b-z)-y,
        x*y-c*z
    ))

def ikeda_attractor(values):
    a, b, c, d = 1, 0.9, 0.4, 6
    x, y, z = values[0], values[1], values[2]

    return np.stack((
        a + b * (x * np.cos(z) - y * np.sin(z)),
        b * (x * np.sin(z) + y * np.cos(z)),
        c - d / (1 + x * x + x * x)
    ))

def chua_attractor(values):
    x, y, z = values[0], values[1], values[2]
    a, b, c, d, e = 15.6, 28, 0, -1.143, -0.714
    g = e*x + (d - e)*(np.abs(x + 1) - np.abs(x - 1)) / 2
    return np.stack((
        a * (y - x - g),
        x - y + z,
        b * y - c * z
    ))

def chua_attractor2(values):
    x, y, z = values[0], values[1], values[2]
    a,b,c,d = 36, 3, 20, -3

    return np.stack((
        a * (y - x),
        x - x * z + c * y + d,
        x * y - b * z
    ))

def many_scroll_chua_attractor(values):
    x, y, z = values[0], values[1], values[2]
    alpha, beta, a, b, c, d = 10.82, 14.286, 1.3, .11, 7, 0

    h = -b * np.sin(np.pi * x / (2*a) + d)
    return np.stack((
        alpha * (y - h),
        x - y + z,
        -1 * beta * y
    ))

def multiscroll_lorenz_attractor(values):
    x, y, z = values[0], values[1], values[2]
    a, b, c = -10, -4, 0
    return np.stack((
        -1 * (a * b) / (a + b) * x - y * z + c,
        a * y + x * z,
        b * z + x * y
    ))

def rossler_attractor(values):
    x, y, z = values[0], values[1], values[2]
    a, b, c = 0.2, 0.2, 5.7
    return np.stack((
        -1 * y - z,
        x + a * y,
        b + z * (x - c)
    ))

def force_at_distance(distances):
    target_distance = 0.001
    # print(np.average(distances))
    repel_strength = 100
    attract_strength = 40
    return np.maximum(0, attract_strength * (distances - target_distance)) + np.minimum(0, repel_strength * (distances - target_distance))

def neighbor_pull(values):
    window_width = 15
    total_pull = np.zeros_like(values)
    for i in range(1, window_width // 2 + 1):
        for p in [-1, 1]:
            difference = np.roll(values, p * i, axis=1) - values
            norm = np.linalg.norm(difference, axis=0)
            total_pull += difference * force_at_distance(norm) / i * window_width
    # values_out = strength * (neighbor_averages(values) - values)
    # magnitudes = np.sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2])
    # return values_out * (magnitudes - 155) / magnitudes
    return total_pull

# def energy(positions, velocities):
#     return velocities * velocities +

class Automaton:
    def __init__(self, state_width=50):
        # self.state = np.stack((
        #     np.linspace(-1, 1, state_width),
        #     np.linspace(-1, 1, state_width),
        #     np.linspace(-1, 1, state_width)
        # ))
        self.state_width = state_width
        seed = 103
        self.attractor = multiscroll_lorenz_attractor
        start_range = (-5, 5)
        noise_scale = 50 # larger scale means smaller features
        self.state = np.stack((
            np.array([scaled_raw_noise_2d(*start_range, seed*10000, t) for t in np.linspace(0, noise_scale, state_width)]),
            np.array([scaled_raw_noise_2d(*start_range, seed*20000, t) for t in np.linspace(0, noise_scale, state_width)]),
            np.array([scaled_raw_noise_2d(*start_range, seed*30000, t) for t in np.linspace(0, noise_scale, state_width)])
        ))
        self.state = np.array([[1], [1], [-1]]) * np.ones_like(self.state)

        zrange = (-1, 1)
        # self.state[2] = np.arange(*zrange, (zrange[1] - zrange[0]) / state_width)

        self.velocity = np.stack((np.zeros(state_width), np.zeros(state_width), np.zeros(state_width)))

    def step(self, timestep):
        # self.state = blur(self.state, ratio=0.0001)
        self.neighbor_pull_strength = 0*0.0001
        self.attractor_strength = 0.1
        self.pull = neighbor_pull(self.state)
        # self.pull = np.zeros_like(self.state)
        self.velocity = self.attractor_strength * self.attractor(self.state) + self.neighbor_pull_strength * self.pull
        # print((np.linalg.norm(attractor(self.state)),np.linalg.norm(neighbor_pull(self.state))))
        # self.velocity -= timestep * timestep * (self.state - attractor(self.state))
        # print(np.linalg.norm(self.state - attractor(self.state)))
        # print(np.max(self.velocity))
        self.state = self.state + timestep * self.velocity


if __name__ == "__main__":
    pass
