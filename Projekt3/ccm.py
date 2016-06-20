#!/usr/bin/python

import argparse as ap
import gzip as gz
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage.filters import gaussian_filter


class CCM(object):
    def __init__(self, input_file, chr_id, chr_len, seg_num, max_iter, max_eps,
                 sub_coords, gauss_range, scale_type, color_map, resize_factor,
                 out_file):
        """
        Creates new chromosomal contact matrix object
        :param input_file: filename for input of matrix
        :param chr_id:  id of chromosime
        :param chr_len:  length of chromosome
        :param seg_num:  length of each segment
        :param max_iter: maximum number of iterations
        :param max_eps:  maximum value of epsilon
        :param sub_coords:  coordinates of submatrix
        :param gauss_range:  range of gaussian filter
        :param scale_type:  type of scale for heatmap
        :param color_map:  colormap for heatmap
        :param resize_factor: resize factor for output matrix
        :param out_file: filename for output of matrix
        """
        self.input_file = input_file
        self.chr_id = chr_id
        self.chr_len = chr_len
        self.seg_num = seg_num
        self.max_iter = max_iter
        self.max_eps = max_eps
        self.sub_coords = sub_coords
        self.gauss_range = gauss_range
        self.scale_type = scale_type
        self.color_map = color_map
        self.resize_factor = resize_factor
        self.out_file = out_file

    @staticmethod
    def parse_arguments():
        """
        Reads command-line arguments
        :return: namespace of arguments for chromosomal contact matrix
                initialization
        """
        parser = ap.ArgumentParser()
        parser.add_argument('-i', '--input_file', required=True, type=str,
                            help='Path to .gz file with chromosomal contacts '
                                 'data')
        parser.add_argument('-ci', '--chr_id', required=True, type=str,
                            help='ID of chromosome in the input file')
        parser.add_argument('-cl', '--chr_len', type=int, default=0,
                            help='Length of chromosome')
        parser.add_argument('-sn', '--seg_num', required=True, type=int,
                            help='Amount of segments the chromosome should be '
                                 'divided into')
        parser.add_argument('-mi', '--max_iter', type=int, default=3,
                            help='Maximum number of normalization iterations')
        parser.add_argument('-me', '--max_eps', type=float, default=0.01,
                            help='Biggest error of regression')
        parser.add_argument('-sc', '--sub_coords', type=int, default=0,
                            nargs='+',
                            help='Coordinates for submatrix that should be '
                                 'displayed. Order of coordiantes is: \
                            left x, right x, top y, bottom y')
        parser.add_argument('-gr', '--gauss_range', type=float, default=0.0,
                            help='Range of gauss filter normalizing output '
                                 'matrix before display')
        parser.add_argument('-st', '--scale_type', required=True,
                            choices=['log', 'linear'],
                            help='Type of scale of displayed heat map.')
        parser.add_argument('-cm', '--color_map', type=str, default='spectral',
                            help='Name of matplotlib colormap for heat map')
        parser.add_argument('-rf', '--resize_factor', type=int, default=1,
                            help='How many times smaller the output matrix '
                                 'should be')
        parser.add_argument('-of', '--out_file', type=str, default=None,
                            help='Filename for output file with matrix')
        return parser.parse_args()

    @staticmethod
    def create_ccm_from_args():
        """
        Creates new CCM object based on command-line arguments
        :return: CCM object
        """
        args = CCM.parse_arguments()
        new = CCM(args.input_file, args.chr_id, args.chr_len, args.seg_num,
                  args.max_iter, args.max_eps,
                  args.sub_coords, args.gauss_range, args.scale_type,
                  args.color_map, args.resize_factor)
        return new

    @staticmethod
    def normalize_matrix_columns(matrix):
        """
        Normalizes all columns of matrix to one(sum of all elements in column
        is equal one)
        :param matrix: matrix to normalize
        :return: normalized numpy matrix
        """
        new = matrix.copy()
        col_sums = new.sum(axis=0)
        new /= col_sums[np.newaxis, :]
        return new

    @staticmethod
    def normalize_matrix_rows(matrix):
        """
            Normalizes all columns of matrix to one(sum of all elements in row
            is equal one)
            :param matrix: matrix to normalize
            :return: normalized numpy matrix
            """
        new = matrix.copy()
        row_sums = new.sum(axis=1)
        new /= row_sums[:, np.newaxis]
        return new

    def parse_data(self):
        """
        Reads data from file and fills chromosomal contact matrix for specified
        chromosome
        """

        # If length of chromosome is not given, we have to calculate it first
        if self.chr_len == 0:
            with gz.open(self.input_file, 'r') as f:
                max_len = 0
                for line in f:
                    row = line.split()
                    first_chr, second_chr = row[0].decode(), row[2].decode()
                    if first_chr == self.chr_id and second_chr == self.chr_id:
                        first_pos, second_pos = int(row[1]), int(row[3])
                        if first_pos > max_len:
                            max_len = first_pos
                        if second_pos > max_len:
                            max_len = second_pos
            self.chr_len = max_len

        self.matrix = np.zeros(((self.chr_len // self.seg_num) + 1,
                                (self.chr_len // self.seg_num) + 1))

        # I assume that input file is gzipped text file
        with gz.open(self.input_file, 'r') as f:
            for line in f:
                row = line.split()
                first_chr, second_chr = row[0].decode(), row[2].decode()
                first_pos, second_pos = int(row[1]), int(row[3])
                if first_chr == self.chr_id and second_chr == self.chr_id:
                    self.matrix[first_pos // self.seg_num][
                        second_pos // self.seg_num] += 1
                    self.matrix[second_pos // self.seg_num][
                        first_pos // self.seg_num] += 1

        # Inter-segment contacts should be omitted
        for i in range(len(self.matrix)):
            self.matrix[i][i] = 0

        # Final matrix should be normalized
        self.matrix = CCM.normalize_matrix_columns(ccm.matrix)
        self.matrix = CCM.normalize_matrix_rows(ccm.matrix)

    @staticmethod
    def get_submatrix(matrix, coords):
        """
        Returns submatrix based on given coordinates
        :param matrix: original matrix
        :param coords: coordinates in form [left x, right x, top y, bot y]
        :return: numpy array that is submatrix of given matrix
        """
        if coords == 0:
            return matrix
        try:
            x_start, x_end, y_start, y_end = coords
        except:
            raise ValueError('Invalid amount of sub_matrix coordinates')

        try:
            x_len = x_end - x_start + 1
            y_len = y_end - y_start + 1
            temp = np.zeros((x_len, y_len))
            for i in range(x_len):
                for j in range(y_len):
                    temp[i][j] = matrix[x_start + i][y_start + j]
        except:
            raise ValueError('Invalid coordinates.')

        return temp

    @staticmethod
    def smoothen_matrix(range, matrix):
        """
        Filters matrix using gaussian filter
        :param range: range of gaussian filter
        :param matrix: matrix to filter
        :return: filtered matrix
        """
        return gaussian_filter(matrix, range)

    @staticmethod
    def get_avg(x, y, size, matrix):
        """
        Retruns average of all elements in given part of matrix
        :param x: x start coordinate (top left)
        :param y: y start coordinate (top left)
        :param size: dimension of submatrix
        :param matrix: original submatrix
        :return: average of values in submatrix
        """
        total = 0
        for i in range(size):
            for j in range(size):
                total += matrix[x + i][y + j]
        return total / (size ** 2)

    def resize_matrix(self):
        """
        Returns resized matrix by specified factor
        :return: new numpy matrix that contains average values of original
        """
        temp = np.zeros((self.matrix.shape[0] // self.resize_factor,
                         self.matrix.shape[0] // self.resize_factor))
        print(temp.shape)
        for i in range(len(temp)):
            for j in range(len(temp)):
                temp[i][j] = CCM.get_avg(i * self.resize_factor,
                                         j * self.resize_factor,
                                         self.resize_factor, self.matrix)
        return temp

    def normalize(self):
        """
        Performs normalization of matrix (minimizes difference between
        M and b*T)
        :return: vector with weights and new normalized matrix
        """
        b = []
        for i in range(len(ccm.matrix)):
            total = 0
            for j in range(len(ccm.matrix)):
                total += ccm.matrix[i][j] ** 2
            b.append(total)

        new_mat = np.zeros((self.matrix.shape[0], self.matrix.shape[1]))
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                new_mat[i][j] = self.matrix[i][j] / (b[i] * b[j])
        new_mat = CCM.normalize_matrix_columns(new_mat)
        new_mat = CCM.normalize_matrix_rows(new_mat)

        iteration, eps = 1, self.max_eps

        while iteration < self.max_iter and eps >= self.max_eps:
            eps = 0
            tots = []
            for i in range(len(new_mat)):
                tot = 0
                for j in range(len(new_mat)):
                    new_eps = self.matrix[i][j] - b[i] * b[j] * new_mat[i][j]
                    tot += new_eps ** 2
                    if new_eps ** 2 > eps:
                        eps = new_eps ** 2
                tots.append(tot)
            eps = max(tots)
            if eps >= self.max_eps:
                for i in range(len(tots)):
                    b[i] /= tots[i]
                for i in range(len(new_mat)):
                    for j in range(len(new_mat)):
                        new_mat[i][j] *= b[i] * b[j]
                new_mat = CCM.normalize_matrix_columns(new_mat)
                new_mat = CCM.normalize_matrix_rows(new_mat)
            iteration += 1
        return b, new_mat

    def show_matrix(self, matrix):
        """
        Creates heatmap of matrrix
        :param matrix: numpy matrix
        """
        if self.scale_type == 'linear':
            im = plt.imshow(matrix, cmap=self.color_map)
            plt.colorbar(im)
        else:
            im = plt.imshow(matrix, cmap=self.color_map,
                            norm=LogNorm(vmin=0.01, vmax=1))
            plt.colorbar(im, ticks=[0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1])

    def save_matrix(self, matrix):
        """
        Saves matrix to file
        :param matrix:  matrix to save
        """
        if self.out_file is not None:
            np.save(self.out_file, matrix)


if __name__ == '__main__':
    ccm = CCM.create_ccm_from_args()
    ccm.parse_data()
    weights, new_mat = ccm.normalize()
    res = CCM.get_submatrix(new_mat, ccm.sub_coords)
    res = ccm.resize_matrix()
    res = CCM.smoothen_matrix(ccm.gauss_range, res)
    ccm.save_matrix(res)
    ccm.show_matrix(res)
    plt.show()
