from cmath import *
import math
import numpy
import pylab as plt
import colorsys
import random
from multiprocessing import Pool
import scipy
from simplexnoise import *

def polarToRGB(r, theta):
    return [float(item) for item in colorsys.hsv_to_rgb(theta / (2 * math.pi) + math.pi, r if r < 1 else 1, r if r < 1 else 1)]

def rectToRGB(c):
    return polarToRGB(abs(c), phase(c))

def derivative(value_list):
    return numpy.array([
        (
            value_list[(n+1) % len(value_list)] -
            value_list[(n-1) % len(value_list)]
        ) / 2
        for n in range(len(value_list))]
    )

def neighborAverages(values):
    averages = [values[0][-1] +
                #values[0][0] +
                values[0][1]]
    for index in range(1, len(values[0]) - 1):
        averages.append(values[0][index - 1]
                        #+ values[0][index]
                        + values[0][index + 1])
    averages.append(values[0][0] +
                    #values[0][-1] +
                    values[0][-2])
    return averages

def unitNoiseAt(x, y, noise_seed):
    return scaled_octave_noise_3d(1, 0.5, 1, -1, 1, x, y, noise_seed)

def generateTransitionTable(state_index, step_index, noise_seed):
    return numpy.array([[unitNoiseAt(x, y, noise_seed) + unitNoiseAt(x, y, noise_seed+1000) * 1.0j for x in numpy.arange(0, 1, 1.0 / resolution)] for y in numpy.arange(0, 1, 1.0 / resolution)])

def getTransition(value, noise_seed):
    #return unitNoiseAt(polar(value)[0], polar(value)[1] / 2 / math.pi, noise_seed) + 1.0j * unitNoiseAt(polar(value)[0], polar(value)[1], noise_seed + 10000)
    return unitNoiseAt(value.real, value.imag, noise_seed) + 1.0j * unitNoiseAt(value.real, value.imag, noise_seed + 10000)

state_width = 80
steps = 400
noise_seed = random.randrange(0, 1000)
while True:

    noise_seed += 10.001
    state = numpy.array([rect(1, theta) for theta in
                         [scaled_octave_noise_2d(3, 0.5, 1, 0, 2 * math.pi, noise_seed, y) for y in numpy.arange(0, 1, 1.0 / state_width)]])
    state_derivatives = derivative(state)
    #slope = numpy.array([getTransition(state_derivatives[i], noise_seed) * state_width / (2 * math.pi) for i in range(len(state_derivatives))])

    momentum = numpy.array([0.0 for cell in state])

    #state = numpy.array([rect(1, theta) for theta in numpy.arange(0, 2*math.pi, 2 * math.pi / state_width)])
    max_state = abs(max(state))
    #scaled_state = [item / max_state for item in state]
    #states = numpy.tile(scaled_state, (steps, 1))
    states = numpy.array([state])


    # slopes = numpy.array([slope])
    for row_index in range(steps):
        print("step " + str(row_index))
        prev_state = state
        state_derivatives = derivative(state)
        neighbor_averages = neighborAverages(states)
        updated = numpy.array([getTransition(state_derivatives[i], noise_seed) * state_width for i in range(len(neighbor_averages))])
        momentum = momentum + state_derivatives
        state = (1 - 0.1) * state + updated * 0.1
        polar_state = [polar(item) for item in state]
        state = [rect(item[0], item[1]) for item in polar_state]
        state = numpy.array([item if abs(item) < 1 else item / abs(item) for item in state])
        states = numpy.append(states, [state], axis=0)
        if sum(abs(state - prev_state)) < 0.1:
            break
        #slopes = numpy.append(slopes, [updated], axis=0)


    #state = numpy.reshape(state, [1, state_width])
    #states = numpy.tile(state, (steps, 1))
    state_image = [[rectToRGB(item) for item in row] for row in states]

    #
    #
    # max_slope = abs(max(slope))
    # #scaled_slope = [item / max_slope for item in slope]
    # #slopes = numpy.tile(scaled_slope, (steps, 1))
    # # slope = numpy.reshape(slope, [1, state_width])
    # # slopes = numpy.tile(slope, (steps, 1))
    #
    # slope_image = [[rectToRGB(item) for item in row] for row in slopes]



    transition_table = numpy.array([[getTransition(x + y * 1j, noise_seed) for x in numpy.arange(0, 1, 1.0 / 100)] for y in numpy.arange(0, 1, 1.0 / 100)])
    transition_image = [[rectToRGB(item) for item in row] for row in transition_table]
    plt.clf()
    plt.subplot(1, 1, 1)
    plt.imshow(state_image, aspect='auto')
    # plt.subplot(3, 1, 2)
    # #plt.imshow(slope_image, aspect='auto')
    # plt.subplot(3, 1, 3)
    # plt.imshow(transition_image, aspect='auto')


    plt.ion()


    plt.draw()
    plt.pause(0.01)
while True:
    plt.pause(11)