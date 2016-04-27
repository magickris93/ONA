import time
import numpy as np
import random as r


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
            self.map[0][0] = self.size * r.uniform(0.5, 1.5)
            self.map[0][self.size - 1] = self.size * r.uniform(0.5, 1.5)
            self.map[self.size - 1][0] = self.size * r.uniform(0.5, 1.5)
            self.map[self.size - 1][self.size - 1] = self.size * \
                                                        r.uniform(0.5, 1.5)
        else:
            self.map = matrix

    def get_value(self, coords):
        return self.map[coords[0]][coords[1]]

    def set_value(self, coords, value):
        self.map[coords[0]][coords[1]] = value

    def get_avg(self, coords):
        x = [self.get_value(v) for v in coords]
        return sum(x) / len(x)

    def get_level(self, level):
        """
        Creates list of squares on specified depth
        :param level: Level of depth | 0 <= level <= self.rank
        :return: List of upper-left corners of all squares on specified level
        """
        res = []
        for i in range(0, self.size - 1, 2 ** level):
            for j in range(0, self.size - 1, 2 ** level):
                res.append((i, j))
        return res

    def get_square(self, upper_left, distance):
        """
        Creates a square vertices coordinates based on initial point(
        upper-left corner) and length
        of a side.
        :param upper_left: Coordinates of upper-left corner of square
        :param distance: Length of square's edge
        :return: List of vertices of square in order : upper left, upper right,
        lower right, lower left
        """
        return [upper_left,
                (upper_left[0], upper_left[1] + distance),
                (upper_left[0] + distance, upper_left[1] + distance),
                (upper_left[0] + distance, upper_left[1])]

    def neighbours(self, point, distance):
        """
        Creates list of coordinates of four neighbours of point in specified
        distance
        :param point: Coordinates of center point
        :param distance: Distance from point
        :return: List of vertices of point's neighbours in order : up, right,
        down, left
        """
        return [((point[0] - distance) % (self.size - 1), point[1]),
                (point[0], (point[1] + distance) % (self.size - 1)),
                ((point[0] + distance) % (self.size - 1), point[1]),
                (point[0], (point[1] - distance) % (self.size - 1))]

    def diamond_step(self, coords, level):
        """
        Performs diamond step on specified square
        :param coords: Square vertices coordinates in order: upper left,
        upper right, lower right, lower left
        :param level: Depth of square
        """
        mid = ((coords[0][0] + coords[2][0]) / 2,
               (coords[0][1] + coords[2][1]) / 2)

        v = self.get_avg(coords) + 2 ** level * self.sigma * np.random.normal()

        self.set_value(mid, v)

    def square_step(self, coords, level):
        """
        Performs upper half of square step on specified square
        :param coords: Square vertices coordinates in order: upper left,
        upper right, lower right, lower left
        :param level: Depth of Square
        """
        up = (coords[0][0], (coords[0][1] + coords[1][1]) / 2)
        lt = ((coords[0][0] + coords[3][0]) / 2, coords[0][1])

        dist = up[1] - up[0]

        v_up = self.get_avg(self.neighbours(up, dist)) + \
               2 ** (level - 1) * self.sigma * np.random.normal()

        v_lt = self.get_avg(self.neighbours(lt, dist)) + \
               2 ** (level - 1) * self.sigma * np.random.normal()

        self.set_value(up, v_up)
        self.set_value(lt, v_lt)

    def diamond_square(self, level=None):
        """
        Performs diamond square algorithm on the map - fills matrix with
        values corresponding to landscape heights.
        """
        if level is None:
            start = self.rank
        else:
            start = level
        for k in range(start, 0, -1):
            squares = self.get_level(k)
            for square in squares:
                self.diamond_step(self.get_square(square, 2 ** k), k)
            for square in squares:
                self.square_step(self.get_square(square, 2 ** k), k)
        for i in range(self.size - 1):
            self.map[self.size - 1][i] = self.map[0][i]
            self.map[i][self.size - 1] = self.map[i][0]
        self.map[self.size - 1][self.size - 1] = self.map[0][0]

    def generate_heatmap(self):
        pass

    def generate_plot(self):
        pass

    def print_map(self):
        print(self.map)


land = Landscape(n=8)
t0 = time.time()
land.diamond_square()
print('Execution time:', time.time() - t0)
land.print_map()
