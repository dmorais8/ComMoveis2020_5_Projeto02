# coding: utf-8
__author__ = "David Morais"

import numpy as np
import random

# CONSTANTS
PI = np.pi
M = 50

if __name__ == '__main__':

    phi_k = 2 * PI * np.random.rand()
    phi_j = 2 * PI * np.random.rand()

    m = np.arange(0, 50, 1)
    x_k = np.sin(4 * PI * m / 5 + phi_k)
    n = 1
    x_j_1 = np.sin(4 * PI * m / 5 + 2 * PI * m * n / M + phi_j)
    n = 2
    x_j_2 = np.sin(4 * PI * m / 5 + 2 * PI * m * n / M + phi_j)
    n = 3
    x_j_3 = np.sin(4 * PI * m / 5 + 2 * PI * m * n / M + phi_j)

    Sum1 = np.sum(x_k * x_j_1)
    print(f'O resultado para n=1 eh: {Sum1}')
    Sum2 = np.sum(x_k * x_j_2)
    print(f'O resultado para n=2 eh: {Sum2}')
    Sum3 = np.sum(x_k * x_j_3)
    print(f'O resultado para n=3 eh: {Sum3}')
