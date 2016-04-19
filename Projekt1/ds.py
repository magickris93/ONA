import numpy as np


class Landscape:
    def __init__(self, matrix=None, n=10, sigma=0.1, colorfile=None,
                 hmfile=None, plotfile=None):
        self.rank = n
        self.size = 2 ** n + 1
        self.sigma = sigma
        self.heatmap_color = colorfile
        self.heatmap_file = hmfile
        self.plot_file = plotfile
        if matrix is None:
            self.map = np.zeros((self.size, self.size))
        else:
            self.map = None

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

    def elevate(self, level, start):
        pass
