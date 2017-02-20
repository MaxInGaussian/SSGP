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

def load_boston_data(proportion=0.1):
    from sklearn import datasets
    from sklearn import cross_validation
    boston = datasets.load_boston()
    X, y = boston.data, boston.target
    y = y[:, None]
    X = X.astype(np.float64)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y, test_size=proportion)
    return X_train, y_train, None, None, X_test, y_test

samples_per_m = 5
M = [50, 100, 150, 200]
penalties = [None, ['ridge', 0.01], ['lasso', 0.00002], ['bridge', 0.8, 0.01]]
MSE = [[] for p in penalties]
NMSE = [[] for p in penalties]
MNLP = [[] for p in penalties]
TIME = [[] for p in penalties]
for m in M:
    sum_mse, sum_nmse = [0]*len(penalties), [0]*len(penalties)
    sum_mnlp, sum_time = [0]*len(penalties), [0]*len(penalties)
    for _ in range(samples_per_m):
        X_train, y_train, X_valid, y_valid, X_test, y_test = \
            load_boston_data()
        for i, penalty in enumerate(penalties):
            start_time = time.time()
            ssgp = SSGP(m, True)
            ssgp.fit(X_train, y_train, SMORMS3(ssgp, penalty))
            elapsed_time = time.time() - start_time
            mse, nmse, mnlp = ssgp.predict(X_test, y_test)
            sum_mse[i] += mse
            sum_nmse[i] += nmse
            sum_mnlp[i] += mnlp
            sum_time[i] += elapsed_time
            print("M is", m, ",", "MSE =", mse)
            print("M is", m, ",", "NMSE =", nmse)
            print("M is", m, ",", "MNLP =", mnlp)
            print("M is", m, ",", "Training Time =", elapsed_time)
    for i, penalty in enumerate(penalties):
        MSE[i].append(sum_mse[i]/samples_per_m)
        NMSE[i].append(sum_nmse[i]/samples_per_m)
        MNLP[i].append(sum_mnlp[i]/samples_per_m)
        TIME[i].append(sum_time[i]/samples_per_m)
import matplotlib.pyplot as plt
f, axarr = plt.subplots(2, 2)
maxv, minv = 0, 1e5
for i, penalty in enumerate(penalties):
    for j in range(len(M)):
        maxv = max(maxv, MSE[i][j])
        minv = min(minv, MSE[i][j])
    axarr[0, 0].plot(M, MSE[i], label=('None' if penalty is None else '-'.join(map(str, penalty))))
axarr[0, 0].set_autoscaley_on(False)
axarr[0, 0].set_ylim([minv-(maxv-minv)*0.25,maxv+(maxv-minv)*0.95])
axarr[0, 0].set_title('MSE vs M')
legend = axarr[0, 0].legend(loc='upper center', shadow=True)
maxv, minv = 0, 1e5
for i, penalty in enumerate(penalties):
    for j in range(len(M)):
        maxv = max(maxv, NMSE[i][j])
        minv = min(minv, NMSE[i][j])
    axarr[0, 1].plot(M, NMSE[i], label=('None' if penalty is None else '-'.join(map(str, penalty))))
axarr[0, 1].set_autoscaley_on(False)
axarr[0, 1].set_ylim([minv-(maxv-minv)*0.25,maxv+(maxv-minv)*0.95])
axarr[0, 1].set_title('NMSE vs M')
legend = axarr[0, 1].legend(loc='upper center', shadow=True)
maxv, minv = 0, 1e5
for i, penalty in enumerate(penalties):
    for j in range(len(M)):
        maxv = max(maxv, MNLP[i][j])
        minv = min(minv, MNLP[i][j])
    axarr[1, 0].plot(M, MNLP[i], label=('None' if penalty is None else '-'.join(map(str, penalty))))
axarr[1, 0].set_autoscaley_on(False)
axarr[1, 0].set_ylim([minv-(maxv-minv)*0.25,maxv+(maxv-minv)*0.95])
axarr[1, 0].set_title('MNLP vs M')
legend = axarr[1, 0].legend(loc='upper center', shadow=True)
maxv, minv = 0, 1e5
for i, penalty in enumerate(penalties):
    for j in range(len(M)):
        maxv = max(maxv, TIME[i][j])
        minv = min(minv, TIME[i][j])
    axarr[1, 1].plot(M, TIME[i], label=('None' if penalty is None else '-'.join(map(str, penalty))))
axarr[1, 1].set_autoscaley_on(False)
axarr[1, 1].set_ylim([minv-(maxv-minv)*0.25,maxv+(maxv-minv)*0.95])
axarr[1, 1].set_title('TIME vs M')
legend = axarr[1, 1].legend(loc='upper center', shadow=True)
plt.show()