#!/usr/bin/python
import argparse as ap
import gzip as gz
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

class CCM(object):

    def __init__(self, input_file, chr_id, chr_len, seg_num, max_iter, max_eps, 
                 sub_coords, gauss_range, scale_type, color_map, resize_factor):
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

    @staticmethod
    def parse_arguments():
        parser = ap.ArgumentParser()
        parser.add_argument('-i', '--input_file', required=True, type=str,
                            help='Path to .gz file with chromosomal contacts data')
        parser.add_argument('-ci', '--chr_id', required=True, type=str,
                            help='ID of chromosome in the input file')
        parser.add_argument('-cl', '--chr_len', type=int, default=0,
                            help='Length of chromosome')
        parser.add_argument('-sn', '--seg_num', required=True, type=int,
                            help='Amount of segments the chromosome should be divided into')
        parser.add_argument('-mi', '--max_iter', type=int, default=3,
                            help='Maximum number of normalization iterations')
        parser.add_argument('-me', '--max_eps', type=float, default=0.01,
                            help='Biggest error of regression')
        parser.add_argument('-sc', '--sub_coords', type=list, default=[],
                            help='Coordinates for submatrix that should be displayed. Order of coordiantes is: \
                            left x, right x, top y, bottom y')
        parser.add_argument('-gr', '--gauss_range', type=float, default=0.0,
                            help='Range of gauss filter normalizing output matrix before display')
        parser.add_argument('-st', '--scale_type', required=True, choices=['log', 'linear'],
                            help='Type of scale of displayed heat map.')
        parser.add_argument('-cm', '--color_map', type=str, default='viridis',
                            help='Name of matplotlib colormap for heat map')
        parser.add_argument('-rf', '--resize_factor', type=int, default=1,
                            help='How many times smaller the output matrix should be')
        return parser.parse_args()

    @staticmethod
    def create_ccm_from_args():
        args = CCM.parse_arguments()
        new = CCM(args.input_file, args.chr_id, args.chr_len, args.seg_num, args.max_iter, args.max_eps, 
                  args.sub_coords, args.gauss_range, args.scale_type, args.color_map, args.resize_factor)
        return new

    def parse_data(self):
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

        self.matrix = np.zeros(((self.chr_len // self.seg_num) + 1, (self.chr_len // self.seg_num) + 1))

        with gz.open(self.input_file, 'r') as f:
            for line in f:
                row = line.split()
                first_chr, second_chr = row[0].decode(), row[2].decode()
                first_pos, second_pos = int(row[1]), int(row[3])
                if first_chr == self.chr_id and second_chr == self.chr_id:
                    self.matrix[first_pos // self.seg_num][second_pos // self.seg_num] += 1
                    self.matrix[second_pos // self.seg_num][first_pos // self.seg_num] += 1
        
    def get_submatrix(self):
        try:
            x_start, x_end, y_start, y_end = self.sub_coords
        except:
            raise ValueError('Invalid amount of sub_matrix coordinates')

        try:
            x_len = x_end - x_start + 1
            y_len = y_end - y_start + 1
            temp = np.zeros((x_len, y_len))
            for i in range(x_len):
                for j in range(y_len):
                    temp[i][j] = self.matrix[x_start+i][y_start+j]
        except:
            raise ValueError('Invalid coordinates.')

        return temp

    def smoothen_matrix(self, matrix):
        return sp.ndimage.filters.gaussian_filter(matrix, self.gauss_range)

