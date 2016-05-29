#!/usr/bin/python

import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.sparse as sp


class MarkovClusterer(object):
    def __init__(self, file_type, input_file, epsilon, exponent, depth):
        self.file_type = file_type
        self.input_file = input_file
        self.eps = epsilon
        self.exponent = exponent
        self.depth = depth
        if self.file_type == 'txt':
            self.matrix = np.loadtxt(input_file)
        elif self.file_type == 'bin':
            self.matrix = np.load(input_file)
        elif self.file_type == 'mat':
            self.matrix = sio.loadmat(input_file)
        else:
            raise ValueError('Invalid matrix file format')
        self.matrix = self.matrix.astype(float)

    @staticmethod
    def square_matrix(matrix):
        return matrix.dot(matrix)

    def inflate_matrix(self, matrix):
        new = np.power(matrix, self.exponent)
        return MarkovClusterer.normalize_matrix_columns(new)

    @staticmethod
    def normalize_matrix_columns(matrix):
        new = matrix.copy()
        if sp.issparse(new):
            col_sums = np.array(new.sum(axis=0))[0, :]
            row_ind, col_ind = new.nonzero()
            for i in range(len(new.data)):
                new.data[i] /= col_sums[row_ind[i]]
        else:
            col_sums = new.sum(axis=0)
            new /= col_sums[np.newaxis, :]
        return new

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
        parser.add_argument('-e', '--exponent', required=True, type=float,
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

    @staticmethod
    def get_matrix_norm(matrix):
        norm = 0
        if sp.issparse(matrix):
            for element in matrix.data:
                norm += abs(element)
        else:
            for row in matrix:
                for element in row:
                    norm += abs(element)
        return norm

    def mcl(self):
        i = 0
        norms = []
        current = MarkovClusterer.normalize_matrix_columns(self.matrix)
        while i < self.depth and (i == 0 or (norms[-1] > self.eps)):
            tmp = MarkovClusterer.square_matrix(current)
            succ = self.inflate_matrix(tmp)
            norms.append(MarkovClusterer.get_matrix_norm(succ - current))
            current = succ
            i += 2
            print(norms[-1])
        return current, norms


if __name__ == '__main__':
    clust = MarkovClusterer.get_clusterer_from_args()
    out, norms = clust.mcl()
    ans = input('Show matrix plot? [y/n]:')
    if ans == 'y':
        xs = range(0, 2*len(norms), 2)
        plt.plot(xs, norms)
        plt.show()
