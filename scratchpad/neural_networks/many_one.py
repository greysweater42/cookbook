import numpy as np

n = 200
K = 3
eta = 0.1
epochs = 100

w_real = [7, -5, 0]
w = np.random.rand(K)
print(w_real)
print(w)

X = np.round(np.array([np.random.rand(K) for i in range(n)]), 3)
t = np.round((X * np.array(w_real)).sum(axis=1), 3)
t = 1 / (1 + np.exp(-t))

errors = []
for epoch in range(epochs):
    dw = 0
    error = 0
    for i, x in enumerate(X):
        u = sum(w * x)
        y = 1 / (1 + np.exp(-u))
        dw += (y - t[i]) * y * (1 - y) * x
        error += 0.5 * ((t[i] - y) ** 2)
    w -= dw * eta
    errors.append(error)

print(np.round(w, 2))
np.round(errors, 4)
