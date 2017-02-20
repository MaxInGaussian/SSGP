################################################################################
#  SSGP: Sparse Spectrum Gaussian Process
#  Github: https://github.com/MaxInGaussian/SSGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import os, sys
import time
import numpy as np
try:
    from SSGP import *
except:
    print("SSGP is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../../")
    from SSGP import *
    print("done.")

def custom_function(X):
    return np.sin(X)

def load_1d_function_data(true_func):
    N = 500
    X_train = np.vstack((np.random.rand(N//2, 1)*N//25,
                   N//25+N//50+np.random.rand(N//2, 1)*N//25))
    noise = 0.1
    y_train = true_func(X_train)+np.random.randn(X_train.shape[0], 1)*noise
    X_test = np.random.rand(N//2, 1)*N//50+N//25
    y_test = true_func(X_test)+np.random.randn(X_test.shape[0], 1)*noise
    return X_train, y_train, X_test, y_test

def plot_1d_test_function(X_train, y_train, predict_func, true_func=None):
    pts = 500
    errors = [0.13, 0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64, 2.2]
    _X, _y = X_train.copy(), y_train.copy()
    grows = np.ceil(np.sqrt(_X.shape[1]))
    import matplotlib.pyplot as plt
    plt.figure()
    for d in range(_X.shape[1]):
        ax = plt.subplot(grows, grows, d + 1)
        xrng = _X[:, d].max() - _X[:, d].min()
        Xs = np.tile(np.linspace(_X[:, d].min() - 0.3*xrng,
            _X[:, d].max() + 0.3*xrng, pts)[:, None], _X.shape[1])
        mu, std = predict_func(Xs)
        mu = mu.ravel()
        for er in errors:
            ax.fill_between(Xs[:, d], mu - er * std, mu + er * std,
                            alpha=((3 - er)/5.)**1.7, facecolor='blue',
                            linewidth=0.0)
        ax.plot(Xs[:, d], mu[:], 'black')
        if(true_func is not None):
            ax.plot(Xs[:, d], true_func(Xs)[:], 'r--')
        ax.errorbar(_X[:, d], _y.ravel(), fmt='r.', markersize=10)
        yrng = _y.max() - _y.min()
        plt.ylim(_y.min() - 0.5*yrng, _y.max() + 0.5*yrng)
        plt.xlim(Xs[:, d].min(), Xs[:, d].max())
    del _X
    plt.show()

X_train, y_train, X_test, y_test = load_1d_function_data(custom_function)
m = 10
ssgp = SSGP(m)
ssgp.fit(X_train, y_train, trainer=SMORMS3(ssgp, penalty=['lasso', 0.02]))
mse, nmse, mnlp = ssgp.predict(X_test, y_test)
print("when M is", m, ",", "MSE =", mse)
print("when M is", m, ",", "NMSE =", nmse)
print("when M is", m, ",", "MNLP =", mnlp)
plot_1d_test_function(X_train, y_train, ssgp.predict, custom_function)