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
    return np.zeros((batch_size, size, size), dtype=np.float64)



def adadelta(X, Y, alpha=1e-3, epochs=100, bs=10, beta1=0.9, beta2=0.999, epsilon=1e-8):
    thetas = init_theta(X.shape[1], type = 'zeros')
    t = 0
    for epoch in np.arange(epochs):
        t += 1
        G = init_G(X.shape[1], bs)
        G2 = init_G(X.shape[1], bs)
        for x, y in zip(batch(X, bs), batch(Y, bs)):
            y = y.reshape((-1, 1))
            y_pred = np.dot(x, thetas)
            error = y_pred - y
            gradient = error * x
            for i, grad in enumerate(gradient): #loop over batch
                for j in range(grad.size):
                    G[i][j][j] = beta1 * G[i][j][j] + (1 - beta1) * grad[j]
                    print(f"grad[j]: {grad[j]}")
                    print(f"grad[j]**2: {grad[j]**2}")
                    # print(f"G2[i][j][j]: {G2[i][j][j]}")
                    G2[i][j][j] = beta2 * G2[i][j][j] + (1 - beta2) * (grad[j] * grad[j])
                    print(f"G2[i][j][j]: {G2[i][j][j]}")

            G /= (1 - beta1 ** t)
            G2 /= (1 - beta2 ** t)
            correction = np.zeros((bs, X.shape[1]))
            for i, (g,g2) in enumerate(zip(G, G2)):
                g_coef = np.linalg.inv(np.sqrt(g2) + epsilon) * g
                correction[i] = np.dot(g_coef, gradient[i].T) * alpha
            thetas = thetas - correction.sum(0).reshape(-1, 1)
        print(f"epoch: {epoch}\nerror sum: {abs(error).sum()}\
                        \nthetas: {thetas.T}")


def main():
    rows = 10000
    X = np.random.randint(1000, size = (rows, 4))
    X[:, 3] = 1
    theta = np.array([1, 3, 4, 100])
    Y = np.dot(X, theta)
    adadelta(X, Y, alpha=1e-3, epochs=100, bs=10, beta1=0.9, beta2=0.999, epsilon=1e-8)
if __name__ == "__main__":
    main()
