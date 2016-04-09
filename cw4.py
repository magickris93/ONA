import numpy as np


# Zadanie 1 - rozklad LU macierzy

def gauss(a, level=0):
    if level < a.shape[0] - 1:
        for i in range(level + 1, a.shape[0]):
            a[i][level] /= a[level][level]
            for j in range(level + 1, a.shape[0]):
                a[i][j] = a[i][j] - a[level][j] * a[i][level]
        gauss(a, level + 1)


def lu(a):
    b = a.astype('f').copy()
    l = np.zeros(a.shape)
    u = np.zeros(a.shape)
    gauss(b)

    for i in range(a.shape[0]):
        for j in range(a.shape[0]):
            if i < j:
                u[i][j] = b[i][j]
            elif i == j:
                l[i][j] = 1
                u[i][j] = b[i][j]
            else:
                l[i][j] = b[i][j]
    return l, u
