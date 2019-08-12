#!/usr/bin/env python3
# -*- coding: utf8 -*-

'''
Khoảng cách Euclid từ 1 điểm z tới 1 điểm x_i trong tập huấn luyện
'''

from __future__ import print_function
import numpy as np
from time import time   # for comparing running time
d, N = 1000, 1000   # dimention, number of training points
X = np.random.randn(N, d)
z = np.random.randn(d)


# naively compute square distance between two vector
def dist_pp(z, x):
    d = z - x.reshape(z.shape)  # force x and z to have the same dims
    return np.sum(d*d)


# from one point to each point in a set, naive
def dist_ps_naive(z, X):
    N = X.shape[0]
    result = np.zeros((1, N))
    for i in range(N):
        result[0][i] = dist_pp(z, X[i])
    return result


# from one point to each point in a set, fast
def dist_ps_fast(z, X):
    X2 = np.sum(X*X, 1)  # square of 12 norm of each X[i], can be precomputed
    z2 = np.sum(z*z)    # square of 12 norm of z
    return X2 + z2 - 2*X.dot(z)  # z2 can be ignored


Z = np.random.randn(100, d)
# from each point in one set to each point in another set, half fast
def dist_ss_0(Z, X):
    M, N = Z.shape[0], X.shape[0]
    result = np.zeros((M, N))
    for i in range(M):
        result[i] = dist_ps_fast(Z[i], X)
        return result


# from each point in one set to each point in another set, fast
def dist_ss_fast(Z, X):
    X2 = np.sum(X*X, 1) # square of 12 norm of each ROW of X
    Z2 = np.sum(Z*Z, 1) # square of 12 norm of each ROW of Z
    return Z2.reshape(-1, 1) + X2.reshape(1, -1) - 2*Z.dot(X.T)

t1 = time()
D1 = dist_ps_naive(z, X)
print('naive point2set, running time:', time() - t1, 's')

t1 = time()
D2 = dist_ps_fast(z, X)
print('naive point2set, running time:', time() - t1, 's')
print('Result difference:', np.linalg.norm(D1 - D2))

t1 = time()
D3 = dist_ss_0(Z, X)
print('\nhalf fast set2set running time:', time() - t1, 's')
t1 = time()
D4 = dist_ss_fast(Z, X)
print('fast set2set running time:', time() - t1, 's')
print('Result difference:', np.linalg.norm(D3 - D4))