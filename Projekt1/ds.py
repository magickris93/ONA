import numpy as np
import random as rand
import time


class Landscape:
    def __init__(self, matrix=None, n=10, sigma=0.1, colorfile=None,
                 hmfile=None, plotfile=None):
        self.rank = n
        self.size = (2 ** n) + 1
        self.sigma = sigma
        self.heatmap_color = colorfile
        self.heatmap_file = hmfile
        self.plot_file = plotfile
        if matrix is None:
            self.map = np.zeros((self.size, self.size))
        else:
            self.map = matrix

    def get_value(self, coords):
        return self.map[coords[0]][coords[1]]

    def set_value(self, coords, value):
        self.map[coords[0]][coords[1]] = value

    def generate_heatmap(self):
        pass

    def generate_plot(self):
        pass

    def square_step(self, coords):
        pass

    def diamond_step(self, coords):
        pass

    def neighbours(self, point, distance):
        #up, right, down, left
        return [((point[0] - distance) % (self.size - 1), point[1]),
                (point[0], (point[1] + distance) % (self.size - 1)),
                ((point[0] + distance) % (self.size - 1), point[1]),
                (point[0], (point[1] - distance) % (self.size - 1))]

    def elevate(self):
        pass

    def print_map(self):
        print(self.map)


t0 = time.time()
land = Landscape(n=5)
print('Execution time:', time.time()-t0)
