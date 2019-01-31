"""
Batch gradient descent by Time Traveller
https://en.wikipedia.org/wiki/Gradient_descent
Code is only for linear regression, no external gradient libraries are used.
"""

import numpy as np
import pandas as pd

def mse(Y_pred, Y):
    return np.square(Y_pred - Y).sum()

class batch():
    def __init__(self, data, bs):
        self.data = data
        self.n = len(data) // bs
        self.bs = bs
        self.current_batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch < self.n:
            start = self.current_batch * self.bs
            end = start  + self.bs
            self.current_batch += 1
            if isinstance(self.data, pd.DataFrame):
                return self.data.iloc[start : end]
            else:
                return self.data[start : end]
        else:
            raise StopIteration

def init_theta(shape, type = 'random'):
    if type == 'random':
        return np.random.randint(10, size=(shape, 1))
    return np.zeros((shape, 1))


def batch_gd(X, Y, alpha, epochs = 100, bs = 10):
    thetas = init_theta(X.shape[1], type = 'zeros')
    for epoch in np.arange(epochs):
        for x, y in zip(batch(X, bs), batch(Y, bs)):
            y = y.reshape((-1, 1))
            y_pred = np.dot(x, thetas)
            error = y_pred - y
            gradient = error * x
            correction = alpha * gradient.T
            correction = correction.sum(1).reshape(-1, 1)
            thetas = thetas - correction
        print(f"epoch: {epoch}\nerror sum: {abs(error).sum()}\
                        \nthetas: {thetas.T}")



def main():
    rows = 10000
    X = np.random.randint(1000, size = (rows, 4))
    X[:, 3] = 1
    theta = np.array([100, 300, 400, 100])
    Y = np.dot(X, theta)
    batch_gd(X, Y, 1e-7, epochs=100, bs = 10) #It's just SGD for bs = 1

if __name__ == "__main__":
    main()
