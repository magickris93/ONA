#!/usr/bin/python

import argparse as ap

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


class MarkovClusterer(object):
    def __init__(self, format, input_file, epsilon, exponent, depth):
        self.format = format
        self.input_file = input_file
        self.eps = epsilon
        self.exponent = exponent
        self.depth = depth
        if self.format == 'txt':
            self.matrix = np.loadtxt(input_file)
        elif self.format == 'bin':
            self.matrix = np.load(input_file)
        elif self.format == 'mat':
            self.matrix = sio.loadmat(input_file)
        else:
            raise ValueError('Invalid matrix file format')

    def square_matrix(self, matrix):
        return matrix.dot(self.matrix)

    def inflate_matrix(self, matrix):
        new = np.power(matrix, self.exponent)
        return MarkovClusterer.normalize_matrix_columns(new)

    @staticmethod
    def normalize_matrix_columns(matrix):
        for i in range(matrix.shape[0]):
            matrix.T[i] /= np.linalg.norm(matrix.T[i])
        return matrix

    @staticmethod
    def parse_command_line_args():
        """
        Returns namespace of command line arguments with their respective values
        :return: Argument namespace
        """
        parser = ap.ArgumentParser()
        parser.add_argument('-i', '--input', required=True, type=str,
                            help='input file for matrix')
        parser.add_argument('-f', '--format', required=True, type=str,
                            help='format of matrix input/output')
        parser.add_argument('-e', '--exponent', required=True, type=int,
                            help='value of an exponent for matrix inflation')
        parser.add_argument('-d', '--depth', required=True, type=int,
                            help='maximum depth of mcl algorithm')
        parser.add_argument('-eps', '--epsilon', required=True, type=float,
                            help='smallest value of matrix norm')
        return parser.parse_args()

    @staticmethod
    def get_clusterer_from_args():
        args = MarkovClusterer.parse_command_line_args()
        return MarkovClusterer(args.format, args.input, args.epsilon,
                               args.exponent, args.depth)

    def mcl(self):
        i = 0
        norms = []
        current = MarkovClusterer.normalize_matrix_columns(self.matrix)
        while i < self.depth and (i == 0 or (norms[-1] > self.eps)):
            tmp = self.square_matrix(current)
            succ = self.inflate_matrix(tmp)
            norms.append(np.linalg.norm(succ - current))
            current = succ
            i += 1
        return current, norms


if __name__ == '__main__':
    clust = MarkovClusterer.get_clusterer_from_args()
    res, norms = clust.mcl()
    ans = input('Show matrix plot? [Y/N]:')
    if ans == 'Y':
        plt.plot(norms)
        plt.show()
