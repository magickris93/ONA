import numpy as np
import random as rand


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

    def neighbours(self, point, distance):
        #up, right, down, left
        return [((point[0] - distance) % (self.size - 1), point[1]),
                (point[0], (point[1] + distance) % (self.size - 1)),
                ((point[0] + distance) % (self.size - 1), point[1]),
                (point[0], (point[1] - distance) % (self.size - 1))]

    def elevate(self):
        # 1. initiate corner values
        nw = (0, 0)
        sw = (self.size - 1, 0)
        ne = (0, self.size - 1)
        se = (self.size - 1, self.size - 1)

        val = self.rank * rand.uniform(0.9, 1.1)

        self.set_value(nw, val)
        self.set_value(ne, val)
        self.set_value(sw, val)
        self.set_value(se, val)

        sqrs = [(nw, ne, se, sw)]

        for k in range(self.rank, 0, -1):
            new = []
            for sqr in sqrs:
                c = ((sqr[0][0] + sqr[2][0]) / 2, (sqr[0][1] + sqr[2][1]) / 2)
                val = np.mean([self.get_value(point) for point in sqr]) + \
                      2 ** k * self.sigma * np.random.normal()

                self.set_value(c, val)

                rt = (c[0], sqr[2][1])
                up = (sqr[0][0], c[1])
                lt = (c[0], sqr[0][1])
                dn = (sqr[2][0], c[1])

                diff = up[1] - nw[1]


'''
                self.set_value(up, np.mean(
                    list(map(self.get_value, self.neighbours(up, diff)))) +
                               2**(k-1)*self.sigma*np.random.normal())

                self.set_value(dn, np.mean(
                    list(map(self.get_value, self.neighbours(dn, diff)))) +
                               2**(k-1)*self.sigma*np.random.normal())

                self.set_value(rt, np.mean(
                    list(map(self.get_value, self.neighbours(rt, diff)))) +
                               2**(k-1)*self.sigma*np.random.normal())

                self.set_value(lt, np.mean(
                    list(map(self.get_value, self.neighbours(lt, diff)))) +
                               2**(k-1)*self.sigma*np.random.normal())

                new.append((nw, up, c, lt))
                new.append((up, ne, rt, c))
                new.append((c, rt, se, dn))
                new.append((lt, c, dn, sw))
'''
            sqrs = new

    def print_map(self):
        print(self.map)


# 2. perform diamond step
# 3. perform square step
# 4. go deeper

land = Landscape(n=6)
land.elevate()
land.print_map()
