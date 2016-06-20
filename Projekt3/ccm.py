#!/usr/bin/python
import argparse as ap


class CCM(object):

    def __init__(self, input_file, chr_id, chr_len, seg_num, max_iter, max_eps, 
                 submatrix_coordinates, gauss_range, scale_type, color_map, resize_factor):
        self.input_file = input_file
        self.chr_id = chr_id
        self.chr_len = chr_len
        self.seg_num = seg_num
        self.max_iter = max_iter
        self.max_eps = max_eps
        self.submatrix_coordinates = submatrix_coordinates
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
                            help='Coordinates for submatrix that should be displayed')
        parser.add_argument('-gr', '--gauss_range', type=float, default=0.0,
                            help='Range of gauss filter normalizing output matrix before display')
        parser.add_argument('-st', '--scale_type', required=True, choices=['log', 'linear'],
                            help='Type of scale of displayed heat map.')
        parser.add_argument('-cm', '--color_map', type=str, default='viridis',
                            help='Name of matplotlib colormap for heat map')
        parser.add_argument('-rf', '--resize_factor', type=int, default=1,
                            help='How many times smaller the output matrix should be')
        return parser.parse_args()
