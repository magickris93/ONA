import numpy as np


def power_iteration(A, eps):
    x = np.random.rand(A.shape[1])
    e = 1
    while e > eps:
        y_new = A.dot(x)
        x_new = y_new / np.linalg.norm(y_new)
        e = np.linalg.norm(x_new - x)
        x = x_new
    return x


print('Power iteration:')
print(power_iteration(np.array([[1, 2.], [4, 3]]), 0.01))


def rayleigh(A, epsilon, mu, x):
    x /= np.linalg.norm(x)
    y = np.linalg.solve((A - mu * np.eye(A.shape[0])), x)
    lam = np.dot(y, x)
    mu = mu + 1 / lam
    e = np.linalg.norm(y - lam * x) / np.linalg.norm(y)
    while e > epsilon:
        x = y / np.linalg.norm(y)
        y = np.linalg.solve((A - mu * np.eye(A.shape[0])), x)
        lam = np.dot(y, x)
        mu = mu + 1 / lam
        e = np.linalg.norm(y - lam * x) / np.linalg.norm(y)
    return x


print("Rayleigh Quotient Iteration")
print(rayleigh(np.array([[1, 2.], [4, 3]]), 0.01, 2, np.random.rand(2)))

# print(power_iteration(np.array([[-1,2,2],[2,2,-1.],[2,-1,2]]), 0.01))
# To nie zadziala, bo ta macierz ma podwojna wartosc wlasna, ktora jest
# jednoczesnie wartoscia dominujaca

print(rayleigh(np.array([[-1, 2, 2], [2, 2, -1.], [2, -1, 2]]), 0.01, 2,
               np.random.rand(3)))
