import numpy as np

# Zadanie 1

def sin_interp(N):
    for i in range(3, N):
        xs = np.linspace(0, math.pi, 1000)
        z = np.polyfit(xs, np.sin(xs), i)
        p = np.poly1d(z)
        #plot(xs, p(xs), 'ko')
        error = sum([(np.sin(x) - p(x))**2 for x in xs])
        print("N = " + str(i) + '\t' + 'error = ' + str(error))

# Zadanie 2

def amp(xs, ys):
    clf()
    d = np.polyfit(xs, ys, 3)
    p = np.poly1d(d)
    x = np.linspace(0, 4, 2000)
    plot(x, [1]*len(x))
    plot(x, p(x))

