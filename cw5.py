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

print('y = {0} + {1}*exp(-x)'.format(a, b))


# Zadanie 2 - obwody pni


p = []
h = []
v = []

with open('trees-stripped.csv', 'r') as myfile:
    for line in myfile.readlines():
        row = line.replace('\n', '').split(',')
        p.append(float(row[0]))
        h.append(float(row[1]))
        v.append(float(row[2]))

x1 = np.vstack([np.array(p), np.array(h)]).T
y = np.array(v)

x2 = []

for i in range(len(p)):
    x2.append(p[i]*h[i])

x2 = np.vstack([np.array(x2), np.ones(len(x2))]).T

a1, b1 = np.linalg.lstsq(x1, y)[0]
print('Kombinacja liniowa obwodu drzewa i wysokosci:')
print('y = {0}*p + {1}*h'.format(a1, b1))

a2, b2 = np.linalg.lstsq(x2, y)[0]
print('Iloczyn obwodu przez wysokosc:')
print('y = {0}*p*h'.format(a2))

x31 = [a1*p[i] + b1*h[i] for i in range(len(p))]
x32 = [a2*h[i]*p[i] for i in range(len(p))]

x3 = np.vstack([x31, x32]).T

a3, b3 = np.linalg.lstsq(x3, y)[0]

print('Kombinacja liniowa powyzszych:')
print('y = {0}*({1}*p + {2}*h) + {3}*h*p'.format(a3, a1, b1, b3))
