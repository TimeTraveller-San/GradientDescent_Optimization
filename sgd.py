import numpy as np
import pandas as pd

def mse(Y_pred, Y):
    return np.square(Y_pred - Y).sum()

def sgd(X, Y, alpha, epochs = 100):
    thetas = np.random.randint(10, size=(X.shape[1], 1))
    print("shape", thetas.shape)
    for epoch in np.arange(epochs):
        shuffled_indices = np.random.permutation(np.arange(X.shape[0]))
        for index in shuffled_indices:
            xi = X.iloc[index]
            yi = Y.iloc[index]
            yi_pred = np.dot(thetas.T, xi)
            error = yi_pred - yi[0]
            gradient = error * np.array(xi)
            correction = alpha * gradient.T
            thetas = thetas.ravel() - correction
            break
        print(f"epoch: {epoch}\nerror: {error}\nthetas: {thetas}")
    Y_pred = np.dot(X, thetas.T)
    for i in range(10):
        print(Y.iloc[i][0], Y_pred[i])

def main():
    X = np.random.randint(1000, size = (100, 4))
    X[:, 3] = np.ones((1, 100))
    theta = np.array([1, 3, 4, 200])
    Y = np.dot(X, theta)
    print(f"X: {X.shape}\nY: {Y.shape}\ntheta: {theta.shape}")
    X, Y = pd.DataFrame(X), pd.DataFrame(Y)
    sgd(X, Y, 1e-8, epochs=100000)

if __name__ == "__main__":
    main()
