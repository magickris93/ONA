#!/usr/bin/python

class CCM(object):

    def __init__(self, input_file, chr_id, chr_len, seg_len, max_iter, max_eps, 
                 submatrix_coordinates, gauss_range, scale_type, color_map, resize_factor):
        self.input_file = input_file
        self.chr_id = chr_id
        self.chr_len = chr_len
        self.seg_len = seg_len
        self.max_iter = max_iter
        self.max_eps = max_eps
        self.submatrix_coordinates = submatrix_coordinates
        self.gauss_range = gauss_range
        self.scale_type = scale_type
        self.color_map = color_map
        self.resize_factor = resize_factor

