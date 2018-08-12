from cmath import *
import math
import numpy
import pylab as plt
import colorsys
import random

from scipy.stats import kde
from multiprocessing import Pool
import scipy
from simplexnoise import *

def polarToRGB(r, theta):
    return [float(item) for item in colorsys.hsv_to_rgb(theta / (2 * math.pi) + math.pi, r if r < 1 else 1, r if r < 1 else 1)]

def rectToRGB(c):
    return polarToRGB(abs(c), phase(c))

def plotDensity(x, y):
    print("plotting density")
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins = 300
    k = kde.gaussian_kde([x, y])
    print("found kde")
    xi, yi = numpy.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    zi = k(numpy.vstack([xi.flatten(), yi.flatten()]))

    # Make the plot
    print("plotting...")
    plt.pcolormesh(xi, yi, numpy.array([math.log(val) if val != 0 else 0 for val in zi]).reshape(xi.shape))
    print("showing")
#    plt.show()


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

def neighborAverages(values):
    averages = [values[-1] +
                values[0] +
                values[1]]
    for index in range(1, len(values) - 1):
        averages.append(values[index - 1]
                        + values[index]
                        + values[index + 1])
    averages.append(values[0] +
                    values[-1] +
                    values[-2])
    return averages

def unitNoiseAt(x, y, noise_seed):
    return scaled_octave_noise_3d(1, 0.5, 1, -1, 1, x, y, noise_seed)

def generateTransitionTable(state_index, step_index, noise_seed):
    return numpy.array([[unitNoiseAt(x, y, noise_seed) + unitNoiseAt(x, y, noise_seed+1000) * 1.0j for x in numpy.arange(0, 1, 1.0 / resolution)] for y in numpy.arange(0, 1, 1.0 / resolution)])

def getTransition(value, noise_seed):
    #return unitNoiseAt(polar(value)[0], polar(value)[1] / 2 / math.pi, noise_seed) + 1.0j * unitNoiseAt(polar(value)[0], polar(value)[1], noise_seed + 10000)
    return unitNoiseAt(value.real, value.imag, noise_seed) + 1.0j * unitNoiseAt(value.real, value.imag, noise_seed + 10000)

def randomState(state_width, seed):
    random_magnitudes = [scaled_octave_noise_2d(3, 0.5, 1, 0, 1, seed, y) for y in numpy.arange(0, 1, 1.0 / state_width)]
    random_phases = [scaled_octave_noise_2d(3, 0.5, 1, 0, 2 * math.pi, seed, y) for y in numpy.arange(0, 1, 1.0 / state_width)]
    return [rect(random_magnitudes[cell_index], random_phases[cell_index]) for cell_index in range(state_width)]


def step(state, noise_seed):
    #state_derivatives = derivative(state)
    #momentum = numpy.array([0.0 for cell in state])
    #prev_state = state
    state_derivatives = derivative(state)
    #neighbor_averages = neighborAverages(state)
    updated = numpy.array(
        [getTransition(state_derivatives[i], noise_seed) * state_width for i in range(len(state_derivatives))])
    #momentum = momentum + state_derivatives
    state = (1 - 0.0) * state + updated * 0.01
    state = neighborAverages(state)
    polar_state = [polar(item) for item in state]
    state = [rect(item[0], item[1]) for item in polar_state]
    state = numpy.array([item if abs(item) < 1 else item / abs(item) for item in state])
    return state, state_derivatives


state_width = 200
upsample_ratio = 0
steps = 300
state_seed = random.randrange(0, 10000)
transition_seed = random.randrange(0, 10000)
while True:
    print("transition seed: " + str(transition_seed) + " state_seed: " + str(state_seed))
    transition_seed += 10.0
    state_seed += 10.0

    # state = numpy.array([
    #     rect(random.random() * 2 - 1, theta) for theta in
    #         [scaled_octave_noise_2d(3, 0.5, 1, 0, 2 * math.pi, state_seed, y) for y in numpy.arange(0, 1, 1.0 / state_width)]
    # ])
    state = numpy.array(randomState(state_width, state_seed))
    states = numpy.array([state])

    if upsample_ratio > 0:
        upsampled_state = numpy.array([rect(1, theta) for theta in
                             [scaled_octave_noise_2d(3, 0.5, 1, 0, 2 * math.pi, state_seed, y) for y in numpy.arange(0, 1, 1.0 / (upsample_ratio * state_width))]])
        upsampled_states = numpy.array([upsampled_state])

    derivatives = []
    for row_index in range(steps):
        print("step " + str(row_index))
        state, state_derivatives = step(state, transition_seed)
        states = numpy.append(states, [state], axis=0)
        derivatives.append(state_derivatives)
    state_image = [[rectToRGB(item) for item in row] for row in states]

    if upsample_ratio > 0:
        for row_index in range(steps * upsample_ratio):
            print("step " + str(row_index))
            upsampled_state, upsampled_state_derivatives = step(upsampled_state, transition_seed)
            upsampled_states = numpy.append(upsampled_states, [upsampled_state], axis=0)
            # if sum(abs(state - prev_state)) < 0.1:
            #     break

        upsampled_state_image = [[rectToRGB(item) for item in row] for row in upsampled_states]

    #
    #
    # max_slope = abs(max(slope))
    # #scaled_slope = [item / max_slope for item in slope]
    # #slopes = numpy.tile(scaled_slope, (steps, 1))
    # # slope = numpy.reshape(slope, [1, state_width])
    # # slopes = numpy.tile(slope, (steps, 1))
    #
    slope_image = [[rectToRGB(item) for item in row] for row in derivatives]



    transition_table = numpy.array([[getTransition(x + y * 1j, transition_seed) for x in numpy.arange(0, 1, 1.0 / 100)] for y in numpy.arange(0, 1, 1.0 / 100)])
    transition_image = [[rectToRGB(item) for item in row] for row in transition_table]



    plt.clf()
    plt.subplot(2,2, 1)
    plt.imshow(state_image, aspect='auto')

    if upsample_ratio > 0:
        plt.subplot(2,2, 2)
        plt.imshow(upsampled_state_image, aspect='auto')

    plt.subplot(2,2,2)
    flattened_derivatives = numpy.ndarray.flatten(numpy.array(states))
    plotDensity(
        numpy.array([derivative.real for derivative in flattened_derivatives[::100]]),
        numpy.array([derivative.imag for derivative in flattened_derivatives[::100]])
    )
    print("shown")

    plt.subplot(2,2, 3)
    plt.imshow(slope_image, aspect='auto')
    plt.subplot(2,2, 4)
    plt.imshow(transition_image, aspect='auto')



    plt.ion()


    plt.draw()
    plt.pause(0.01)