import graphics
import cmath
import colorsys
import math
from tkinter import *
import numpy as np
import time
import random
from ContinuousAutomaton import ContinuousAutomaton, randomState

def polarToRGB(r, theta):
    return [int(item * 255) for item in colorsys.hsv_to_rgb(theta / (2 * math.pi) + math.pi, r if r < 1 else 1, r if r < 1 else 1)]

def rectToRGB(c):
    return polarToRGB(abs(c), cmath.phase(c))

def RGBToHex(components):
    return '#%02x%02x%02x' % tuple(components)

def HeatmapToRGB(bucket_counts):
    return [[RGBToHex([int(math.sqrt(count))] * 3) for count in row] for row in bucket_counts]

class Display:
    def __init__(self):
        self.scale_multiplier = 2
        #self.win = graphics.GraphWin("My Circle", 600, 300, autoflush=False)
        self.iteration = 0
        self.target_iteration = 400
        self.automaton_width = 200
        self.automaton = ContinuousAutomaton(self.automaton_width)

        self.transition_table_width = 100
        self.element_padding = 50  # How many pixels to put between sections of the display
        self.transition_history = [[0] * self.transition_table_width] * self.transition_table_width

        self.frame_count = 5
        self.scaled_automaton_width = int(self.automaton_width * self.scale_multiplier)
        self.window_width = self.scaled_automaton_width * self.frame_count
        self.window_height = int(self.target_iteration * self.scale_multiplier)
        self.tk_root = Tk()
        self.tk_root.geometry(str(self.window_width) + "x" + str(self.window_height))
        self.root_frame = Frame(self.tk_root)
        self.root_frame.pack(expand = True)
        self.state_frame = Canvas(self.root_frame, width=self.scaled_automaton_width, height=self.window_height)
        self.state_frame.pack(side = LEFT)
        self.transition_frame = Canvas(self.root_frame, width=self.scaled_automaton_width, height=self.window_height)
        self.transition_frame.pack(side = LEFT, expand = True, fill=Y)
        self.updates_frame = Canvas(self.root_frame, width=self.scaled_automaton_width, height=self.window_height)
        self.updates_frame.pack(side=LEFT, expand=True, fill=Y)
        self.transition_table_frame = Canvas(self.root_frame, width=self.transition_table_width * self.scale_multiplier, height=self.window_height)
        self.transition_table_frame.pack(side=LEFT, expand=True, fill=Y)
        self.transition_history_frame = Canvas(self.root_frame, width=self.transition_table_width * self.scale_multiplier, height=self.window_height)
        self.transition_history_frame.pack(side=LEFT, expand=True, fill=Y)

        self.tk_root.bind_all("<Key>", self.keypress_handler)
        #self.tk_root.bind_all("<Button-1>", self.click_handler)

    def reset(self, state_seed=None, transition_seed=None):
        self.iteration = 0
        self.automaton = ContinuousAutomaton(self.automaton_width, state_seed=state_seed, transition_seed=transition_seed)
        self.main()

    def reset_state(self, state_seed=None):
        self.iteration = 0
        if state_seed is None:
            state_seed = random.randrange(0, 10000)
        self.automaton.state = np.array(randomState(self.automaton_width, state_seed))

        self.main()

    def keypress_handler(self, event):
        small_seed_offset = 0.001
        large_seed_offset = 0.1
        print("Keypress")
        if event.keysym == "Right":
            self.reset()
        elif event.keysym == "Left":
            self.reset(state_seed=self.automaton.state_seed)
        elif event.keysym == "Up":
            self.reset_state(self.automaton.transition_seed + large_seed_offset)
        elif event.keysym == "Down":
            self.reset_state(self.automaton.transition_seed + small_seed_offset)


    def click_handler(self, event):
        print("click")
        self.reset()

    def draw_state(self):
        for index, cell in enumerate(self.automaton.state):
            self.state_frame.create_rectangle(
                index * self.scale_multiplier,
                self.iteration * self.scale_multiplier,
                (index + 1) * self.scale_multiplier,
                (self.iteration + 1) * self.scale_multiplier,
                fill=RGBToHex(rectToRGB(cell)),
                outline=""
            )
            #self.state_frame.plot(index, self.iteration, graphics.color_rgb(*rectToRGB(cell)))

    def draw_transition_inputs(self, state_derivatives):
        for index, cell in enumerate(state_derivatives):
            self.transition_frame.create_rectangle(
                index * self.scale_multiplier,
                self.iteration * self.scale_multiplier,
                (index + 1) * self.scale_multiplier,
                (self.iteration + 1) * self.scale_multiplier,
                fill=RGBToHex(rectToRGB(cell)),
                outline=""
            )

    def draw_transition_outputs(self, state_updates):
        for index, cell in enumerate(state_updates):
            self.updates_frame.create_rectangle(
                index * self.scale_multiplier,
                self.iteration * self.scale_multiplier,
                (index + 1) * self.scale_multiplier,
                (self.iteration + 1) * self.scale_multiplier,
                fill=RGBToHex(rectToRGB(cell)),
                outline=""
            )

    def draw_transition_table(self):
        # for x_index, x in enumerate(np.linspace(0, 1, self.transition_table_width)):
        #     for y_index, y in enumerate(np.linspace(0, 2 * math.pi, self.transition_table_width)):
        #         transition_value_rect = self.automaton.getTransition(complex(x, y), self.automaton.transition_seed)
        #         transition_color = RGBToHex(rectToRGB(transition_value_rect))
        #         self.transition_table_frame.create_rectangle(
        #             x_index * self.scale_multiplier,
        #             y_index * self.scale_multiplier,
        #             (x_index + 1) * self.scale_multiplier,
        #             (y_index + 1) * self.scale_multiplier,
        #             fill=transition_color,
        #             outline=""
        #         )
        for r_index, r in enumerate(np.linspace(0, 1, self.transition_table_width)):
            for theta_index, theta in enumerate(np.linspace(-1 * math.pi, 1 * math.pi, int(self.transition_table_width))):
                transition_value_rect = self.automaton.getTransition(cmath.rect(r, theta), self.automaton.transition_seed)
                transition_color = RGBToHex(rectToRGB(transition_value_rect))
                self.transition_history_frame.create_rectangle(
                    theta_index * self.scale_multiplier,
                    r_index * self.scale_multiplier,
                    (theta_index + 1) * self.scale_multiplier,
                    (r_index + 1) * self.scale_multiplier,
                    fill=transition_color,
                    outline=""
                )

    def draw_transition_history(self):
        # for x_index, x in enumerate(np.linspace(0, 1, self.transition_table_width)):
        #     for y_index, y in enumerate(np.linspace(0, 2 * math.pi, self.transition_table_width)):
        #         transition_value_rect = self.automaton.getTransition(complex(x, y), self.automaton.transition_seed)
        #         transition_color = RGBToHex(rectToRGB(transition_value_rect))
        #         self.transition_table_frame.create_rectangle(
        #             x_index * self.scale_multiplier,
        #             y_index * self.scale_multiplier,
        #             (x_index + 1) * self.scale_multiplier,
        #             (y_index + 1) * self.scale_multiplier,
        #             fill=transition_color,
        #             outline=""
        #         )

        heatmap_colors = HeatmapToRGB(self.transition_history)
        for r_index, r in enumerate(np.linspace(0, 1, self.transition_table_width)):
            for theta_index, theta in enumerate(np.linspace(-1 * math.pi, 1 * math.pi, int(self.transition_table_width))):
                #transition_value_rect = self.automaton.getTransition(cmath.rect(r, theta), self.automaton.transition_seed)
                transition_heatmap_color = heatmap_colors[r_index][theta_index]
                self.transition_table_frame.create_rectangle(
                    theta_index * self.scale_multiplier,
                    r_index * self.scale_multiplier,
                    (theta_index + 1) * self.scale_multiplier,
                    (r_index + 1) * self.scale_multiplier,
                    fill=transition_heatmap_color,
                    outline=""
                )

    def main(self):
        self.draw_transition_table()
        self.draw_transition_history()
        while self.iteration < self.target_iteration:
            self.draw_state()

            if self.iteration % 20 == 0:
                self.root_frame.update()
                # if display.win.checkMouse() is not None:
                #     display.reset()

            state, state_derivatives, updated = self.automaton.step()

            for derivative in state_derivatives:
                r_cell_index = int(abs(derivative) * self.transition_table_width)
                theta_cell_index = int((cmath.phase(derivative) + math.pi) * self.transition_table_width / 2 * math.pi)
                print(r_cell_index, theta_cell_index)
                #self.transition_history[]

            self.draw_transition_inputs(state_derivatives)
            self.draw_transition_outputs(updated)
            self.iteration += 1


display = Display()
display.main()
display.tk_root.mainloop()