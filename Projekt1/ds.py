#!/usr/bin/python

import argparse as ap
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Landscape:
    def __init__(self, rank=10, sigma=0.1, color_file='terrain',
                 heatmap_file='heatmap', surf_file='surf_plot',
                 heatmap_format='png', surf_format='png', start=10,
                 matrix_in_file=None, matrix_out_file=None):
        """
        Creates Landscape object
        :param rank: Rank of matrix
        :param sigma: Mountainous coefficient
        :param color_file: File with color maps for plots
        :param heatmap_file: Output file for heatmap
        :param surf_file: Output file for surface plot
        :param heatmap_format: Format of heatmap file
        :param surf_format: Format of surface plot file
        :param start: Initial depth at which diamond square algorithm starts
        :param matrix_in_file: Input file with numpy matrix
        :param matrix_out_file: Output file for numpy matrix
        """
        self.rank = rank
        self.size = 2 ** rank + 1
        self.sigma = sigma

        if start is None:
            self.start = self.rank
        elif start > self.rank or start < 0:
            raise ValueError('Invalid depth value - should be greater or equal'
                             '0 and less or equal to rank of matrix')
        else:
            self.start = start

        self.color_file = color_file
        self.heatmap_file = heatmap_file
        self.surf_file = surf_file
        self.heatmap_format = heatmap_format
        self.surf_format = surf_format
        self.matrix_in_file = matrix_in_file
        self.matrix_out_file = matrix_out_file

        if self.matrix_in_file is None:
            self.map = np.zeros((self.size, self.size))
            self.set_corners()
        else:
            self.load_map()

    @staticmethod
    def parse_command_line_args():
        """
        Returns namespace of command line arguments with their respective values
        :return: Argument namespace
        """
        parser = ap.ArgumentParser()
        parser.add_argument('-r', '--rank', required=True, type=int, default=10,
                            help='rank of matrix (2^n + 1)')
        parser.add_argument('-s', '--sigma', required=True, type=float,
                            help='degree of mountainous character')
        parser.add_argument('-mf', '--map_file', type=str,
                            help='output file for heatmap')
        parser.add_argument('-sf', '--surf_file', type=str,
                            help='output file for surface plot')
        parser.add_argument('-me', '--map_ext', type=str,
                            help='format of output file for heatmap')
        parser.add_argument('-se', '--surf_ext', type=str,
                            help='format of output file for surface plot')
        parser.add_argument('-cf', '--color_file', type=str, default='terrain',
                            help='file with colors for plots')
        parser.add_argument('-m', '--mat_file', type=str,
                            help='input file for matrix')
        parser.add_argument('-o', '--out_file', type=str,
                            help='output file for matrix')
        parser.add_argument('-d', '--depth', type=int,
                            help='starting depth of landscape formation')

        return parser.parse_args()

    @staticmethod
    def get_landscape_from_args():
        """
        Creates Landscape object based on command line arguments
        :return: new Landscape object
        """
        args = Landscape.parse_command_line_args()
        return Landscape(args.rank, args.sigma, args.color_file, args.map_file,
                         args.surf_file, args.map_ext, args.surf_ext,
                         args.depth, args.mat_file, args.out_file)

    def set_corners(self, value=None):
        """
        Sets values of all corners of landscape matrix to a specified value. If
        no value is given it is filled with size of matrix multiplied by a
        number x such that 0.6 < x < 1.4
        """
        if value is None:
            value = self.size * rand.uniform(0.6, 1.4)
        self.set_value((0, 0), value)
        self.set_value((0, self.size - 1), value)
        self.set_value((self.size - 1, 0), value)
        self.set_value((self.size - 1, self.size - 1), value)

    def get_value(self, point):
        """
        Returns value from matrix at given coordinates
        :param point: Tuple representing point coordinates
        :return: Value of matrix at given point
        """
        return self.map[point[0]][point[1]]

    def set_value(self, point, value):
        """
        Changes the value at given coordinates
        :param point: Tuple representing point coordinates
        :param value: New value
        """
        self.map[point[0]][point[1]] = value

    def get_avg(self, points):
        """
        Returns an average of values of points in list
        :param points: List of tuples representing point coordinates
        :return: Average of point values
        """
        x = [self.get_value(vertex) for vertex in points]
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

    @staticmethod
    def get_square(upper_left, distance):
        """
        Creates a square vertices coordinates based on initial point(
        upper-left corner) and length
        of a side.
        :param upper_left: Coordinates of upper-left corner of square
        :param distance: Length of square's edge
        :return: List of vertices of square in order : upper left, upper right,
        lower right, lower left
        """
        return [(upper_left[0], upper_left[1]),
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
        left = ((coords[0][0] + coords[3][0]) / 2, coords[0][1])

        dist = up[1] - coords[0][1]

        value_up = self.get_avg(self.neighbours(up, dist)) + \
                   2 ** (level - 1) * self.sigma * np.random.normal()
        value_left = self.get_avg(self.neighbours(left, dist)) + \
                     2 ** (level - 1) * self.sigma * np.random.normal()

        self.set_value(up, value_up)
        self.set_value(left, value_left)

    def diamond_square(self):
        """
        Performs diamond square algorithm on the map - fills matrix with
        values corresponding to landscape heights.
        """
        for k in range(self.start, 0, -1):
            squares = self.get_level(k)
            for square in squares:
                self.diamond_step(Landscape.get_square(square, 2 ** k), k)
            for square in squares:
                self.square_step(Landscape.get_square(square, 2 ** k), k)
            for i in range(self.size - 1):
                self.map[self.size - 1][i] = self.map[0][i]
                self.map[i][self.size - 1] = self.map[i][0]
            self.map[self.size - 1][self.size - 1] = self.map[0][0]

    def generate_heatmap(self):
        """
        Generates heatmap plot for created landscape based on values in matrix
        :param show: If true shows a plot, otherwise saves to file
        :param output_file: Output file name
        :return: True if there is a plot to show, False otherwise
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(self.map, cmap=self.color_file)
        if self.heatmap_file is not None:
            plt.savefig(self.heatmap_file, format=self.heatmap_format)
            plt.close()

    def generate_plot(self):
        """
        Generates 3d surface plot of created landscape based on values in matrix
        :param show: If true shows a plot, otherwise saves to file
        :param output_file: Output file name
        :return: True if there is a plot to show, False otherwise
        """
        length = self.map.shape[0]
        width = self.map.shape[1]
        x, y = np.meshgrid(np.arange(length), np.arange(width))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(x, y, self.map, cmap=self.color_file,
                        rstride=2 * self.rank, cstride=2 * self.rank)
        if self.surf_file is not None:
            plt.savefig(self.surf_file, format=self.surf_format)
            plt.close()

    def make_plots(self):
        """
        Plots surface and heatmap for matrix of landscape
        """
        self.generate_plot()
        self.generate_heatmap()

    def save_map(self):
        """
        Saves matrix to file with .npy extension (numpy format)
        """
        np.save(self.matrix_out_file, self.map)

    def load_map(self):
        """
        Loads matrix from file of numpy format
        """
        self.map = np.load(self.matrix_in_file)


if __name__ == '__main__':
    land = Landscape.get_landscape_from_args()
    land.diamond_square()
    land.make_plots()
    plt.show()
