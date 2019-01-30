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

#adadelta like adagrad is an adaptive gradient method i.e. different weights
#have different personalized learning rates. adagrad faced exploding and
#vanishing gradient issues which adadelta solves by introducing a decaying
#average. It also removes the need to set a learning rate!

def adadelta(X, Y, epochs = 100, bs = 10, gamma = 0.9, epsilon = 1e-10):
    thetas = init_theta(X.shape[1], type = 'zeros')
    correction_ra =  np.zeros((bs, X.shape[1]))
    for epoch in np.arange(epochs):
        G = init_G(X.shape[1], bs)
        for x, y in zip(batch(X, bs), batch(Y, bs)):
            y = y.reshape((-1, 1))
            y_pred = np.dot(x, thetas)
            error = y_pred - y
            gradient = error * x
            for i, grad in enumerate(gradient): #loop over batch
                for j in range(grad.size): #loop over each weight's gradient
                    #Here, G is resticted to an accumulated window to prevent
                    #gradient explosion and vanishing. Rather than inefficiently
                    #story previous gradients, we define G recursively as
                    #decaying average of all past gradients. In other words,
                    #this is linear interpolation of previous squared grads sum
                    #and current grad square as follows. gamma is decaying
                    #factor generally around 0.9 (in the paper)
                    G[i][j][j] = gamma * G[i][j][j] + (1 - gamma) * (grad[j] ** 2)
            correction = np.zeros((bs, X.shape[1]))
            for i, g in enumerate(G):
                g = np.linalg.inv(np.sqrt(g + epsilon))
                correction[i] = np.dot(g, gradient[i].T)
            correction *= np.sqrt(correction_ra + epsilon)
            thetas = thetas - correction.sum(0).reshape(-1, 1)
            correction_ra = gamma * correction_ra + (1 - gamma) * correction ** 2
        print(f"epoch: {epoch}\nerror sum: {abs(error).sum()}\
                        \nthetas: {thetas.T}")


def main():
    rows = 10000
    X = np.random.randint(1000, size = (rows, 4))
    X[:, 3] = 1
    theta = np.array([1, 3, 4, 100])
    Y = np.dot(X, theta)
    adadelta(X, Y, epochs=100, gamma = 0.9, bs = 10) #It's just SGD for bs = 1

if __name__ == "__main__":
    main()
