# coding: utf-8
__author__ = "David Morais"

# Lib imports
import numpy as np
from matplotlib import pyplot as plt

# CONSTANTS
n_bits = 100            # Número de bits
T = 50                  # Tempo de símbolo
Ts = 2                  # Tempo de símbolo em portadora única
K = T/Ts                # Número de subportadoras independentes
N = 2*K                 # N pontos da IDFT

if __name__ == '__main__':

    data_in = np.random.rand(1, n_bits)[0]
    data_in = np.sign(data_in - .5)
    print()
