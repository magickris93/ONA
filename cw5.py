import numpy as np
import math

# Zadanie 0 - metoda najmniejszych kwadratów (regresja liniowa)

x = np.array([0,1,2])
m = np.vstack([x, np.ones(len(x))]).T

y = np.array([6, 0, 0])

a, b = np.linalg.lstsq(m, y)[0]

# print('y = {0}x + {1}'.format(a, b))


# Zadanie 1 - metoda najmniejszych kwadratów (regresja liniowa) cz. 2 

x = np.array(np.linspace(0, 10, 9))

for i in range(len(x)):
    x[i] = math.exp(-x[i])

m = np.vstack([x, np.ones(len(x))]).T

y = np.array([  4.00000000000000e+00, 3.28650479686019e+00, 3.08208499862390e+00, 
                3.02351774585601e+00, 3.00673794699909e+00, 3.00193045413623e+00, 
                0.00055308437015e+00, 3.00015846132512e+00, 3.00004539992976e+00])

a, b = np.linalg.lstsq(m, y)[0]

# print('y = {0} + {1}*exp(-x)'.format(a, b))


# Zadanie 2 - obwody pni



