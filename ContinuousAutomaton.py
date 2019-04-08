import random
import numpy
from simplexnoise import *
import math
import cmath
from cmath import polar

def derivative(value_list):
    return numpy.array([
        (
                value_list[(n+1) % len(value_list)]-
                value_list[(n-1) % len(value_list)]
        ) / 2
        for n in range(len(value_list))]
    )
    # return numpy.array([
    #     (
    #         (
    #             value_list[(n + 2) % len(value_list)]+
    #             value_list[(n+1) % len(value_list)])
    #         ) - (
    #             value_list[(n-1) % len(value_list)]+
    #             value_list[(n -2) % len(value_list)]
    #     ) / 2
    #     for n in range(len(value_list))]
    #)


def neighborAverages(values, input_residual_proportion = 0):
    averages = [(values[-2] +
                values[1] +
                values[0]) / 3]
    for index in range(1, len(values) - 1):
        averages.append((values[index - 1]
                        + values[index]
                        + values[index + 1]) / 3)
    averages.append((values[0] +
                    values[-1] +
                    values[-2]) / 3)

    return [value * input_residual_proportion + average * (1 - input_residual_proportion) for value, average in zip(values, averages)]


def unitNoiseAt(x, y, noise_seed):
    return scaled_octave_noise_3d(2, 0.5, 1, -1.0, 1.0, x, y, noise_seed)


def randomState(state_width, seed):
    random_magnitudes = [scaled_octave_noise_2d(3, 0.5, 6, 0, 1, seed, y) for y in numpy.arange(0, 1, 1.0 / state_width)]
    random_phases = [scaled_octave_noise_2d(3, 0.5, 6, 0, 2 * math.pi, seed, y) for y in numpy.arange(0, 1, 1.0 / state_width)]
    return [cmath.rect(random_magnitudes[cell_index], random_phases[cell_index]) for cell_index in range(state_width)]


class ContinuousAutomaton:
    def __init__(self, width, state_seed = None, transition_seed = None):
        self.state_width = width
        self.upsample_ratio = 1
        self.state_seed = random.randrange(0, 10000) if state_seed is None else state_seed
        self.transition_seed = random.randrange(0, 10000) if transition_seed is None else transition_seed
        self.state = numpy.array(randomState(self.state_width, self.state_seed))
        self.states = numpy.array([self.state])

    def step(self):
        # state_derivatives = derivative(state)
        #momentum = numpy.array([0.0 for cell in self.state])
        # prev_state = state
        state_derivatives = derivative(self.state)
        #state_derivatives = neighborAverages(state_derivatives)
        # neighbor_averages = neighborAverages(state)
        updated = numpy.array(
            [self.getTransition(state_derivatives[i], self.transition_seed) for i in range(len(state_derivatives))])
        #momentum = momentum + state_derivatives
        updated = numpy.array(neighborAverages(updated, input_residual_proportion=0.5))
        state = (1 - 0.1) * self.state + updated * 0.2
        #state = (1 - 0.05) * self.state
        #state = updated
        #state = neighborAverages(state)
        #polar_state = [cmath.polar(item) for item in state]
        #state = [cmath.rect(item[0] + 0.05, item[1] + 0) for item in polar_state]
        state = numpy.array([item if abs(item) < 1 else item / abs(item) for item in state])
        self.state = state
        return state, state_derivatives, updated

    def getTransition(self, value, noise_seed):
        polar_value = polar(value)
        return unitNoiseAt(polar_value[0], polar_value[1] / 2 / math.pi, noise_seed) + 1.0j * unitNoiseAt(polar_value[0], polar_value[1] / 2 / math.pi, noise_seed + 10000)
        #return unitNoiseAt(value.real, value.imag, noise_seed) + 1.0j * unitNoiseAt(value.real, value.imag, noise_seed + 10000)