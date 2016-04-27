import time

import numpy as np


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

    def get_level(self, level):
        """
        Creates list of squares on specified depth
        :param level: Level of depth | 0 <= level <= self.rank
        :return: List of upper-left corners of all squares on specified level
        """
        res = []
        for i in range(0, self.size-1, 2**level):
            for j in range(0, self.size-1, 2**level):
                res.append((i, j))
        return res

    def get_square(self, upper_left, distance):
        """
        Creates a square vertices coordinates based on initial point(
        upper-left corner) and length
        of a side.
        :param upper_left: Coordinates of upper-left corner of square
        :param distance: Length of square's edge
        :return: Tuple of vertices of square in order : upper left, upper right,
        lower right, lower left
        """
        return (upper_left,
                (upper_left[0], upper_left[1] + distance),
                (upper_left[0] + distance, upper_left[1] + distance),
                (upper_left[0] + distance, upper_left[1]))

    def neighbours(self, point, distance):
        """
        Creates tuple of coordinates of four neighbours of point in specified
        distance
        :param point: Coordinates of center point
        :param distance: Distance from point
        :return: Tuple of vertices of point's neighbours in order : up, right,
        down, left
        """
        return (((point[0] - distance) % (self.size - 1), point[1]),
                (point[0], (point[1] + distance) % (self.size - 1)),
                ((point[0] + distance) % (self.size - 1), point[1]),
                (point[0], (point[1] - distance) % (self.size - 1)))

    def elevate(self):
        pass

    def print_map(self):
        print(self.map)


t0 = time.time()
land = Landscape(n=8)
print('Execution time:', time.time() - t0)
