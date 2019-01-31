"""
adagrad is adaptive gradient i.e. different weights have different personalized
learning rates. It helps in case of sparse features where some require a very
high learning rate while other features requires a smaller learning rate.
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

def init_G(size, batch_size):
    return np.zeros((batch_size, size, size))



def adagrad(X, Y, alpha, epochs = 100, bs = 10, epsilon = 1e-10):
    thetas = init_theta(X.shape[1], type = 'zeros')
    for epoch in np.arange(epochs):
        G = init_G(X.shape[1], bs)
        for x, y in zip(batch(X, bs), batch(Y, bs)):
            y = y.reshape((-1, 1))
            y_pred = np.dot(x, thetas)
            error = y_pred - y
            gradient = error * x
            for i, grad in enumerate(gradient): #loop over batch
                for j in range(grad.size): #loop over each weight's gradient
                    G[i][j][j] += grad[j] * grad[j] #update G (G is a diagnol matrix as per the paper) by adding squares of gradient
            correction = np.zeros((bs, X.shape[1]))
            for i, g in enumerate(G):
                g = np.linalg.inv(np.sqrt(g + epsilon))
                #now multiply this g by gradient of each one in the batch TODO
                correction[i] = np.dot(g, gradient[i].T) * alpha

            # adaptive = np.linalg.inv(np.sqrt(G) + epsilon)
            # adaptive_alpha = adaptive * alpha
            # correction = adaptive_alpha * gradient.T
            correction = correction.sum(0).reshape(-1, 1)
            thetas = thetas - correction
        print(f"epoch: {epoch}\nerror sum: {abs(error).sum()}\
                        \nthetas: {thetas.T}")


def main():
    rows = 10000
    X = np.random.randint(1000, size = (rows, 4))
    X[:, 3] = 1
    theta = np.array([1, 3, 4, 100])
    Y = np.dot(X, theta)
    adagrad(X, Y, 1e-2, epochs=100, bs = 10) #It's just SGD for bs = 1
    #alpha is set to 1e-2 in the original paper.

if __name__ == "__main__":
    main()
