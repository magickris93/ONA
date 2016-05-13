import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft


# Zadanie 1

def generate_signal(points):
    gauss = signal.gaussian(points, std=points * 7 / 50)
    sig = np.repeat([0., 1., 0.], int(points / 3))
    filtered = signal.convolve(sig, gauss, mode='same') / sum(gauss)
    plt.plot(gauss, 'b-')
    plt.plot(sig, 'r-')
    plt.plot(filtered, 'k--')
    plt.show()


# generate_signal(10000)

# Zadanie 2

def read_signal_to_array(filename, amount):
    res = []
    with open(filename, 'r') as f:
        for i in range(amount):
            res.append(f.readline().rstrip('\n'))
    return np.array(res)


def show_signal(filename, amount):
    x = np.array(range(amount))
    y = read_signal_to_array('chr1.txt', amount)
    plt.plot(x, y, 'r-')
    ft = fft(y)
    for i in range(int(len(ft) / 10), len(ft)):
        ft[i] = 0
    ift = ifft(ft)
    # plt.plot(x, ft, 'b-')
    plt.plot(ift, 'k--')
    plt.show()


show_signal('chr1.txt', 1000)

