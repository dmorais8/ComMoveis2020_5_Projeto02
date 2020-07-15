# coding: utf-8
__author__ = "David Morais"

# Lib imports
import numpy as np
from matplotlib import pyplot as plt

# CONSTANTS
n_bits = 1000            # Número de bits
T = 500                  # Tempo de símbolo
Ts = 2                  # Tempo de símbolo em portadora única
K = T/Ts                # Número de subportadoras independentes
N = 2*K                 # N pontos da IDFT
PI = np.pi
sigmas = [0, 0.1, 1]
i = 1j

if __name__ == '__main__':

    data_in = np.random.rand(1, n_bits)
    data_in = np.sign(data_in - .5)
    data_in_matrix = data_in.reshape((n_bits//4, 4))

    seq16qam = 2 * data_in_matrix[:, 0] + data_in_matrix[:, 1] + i * (2 * data_in_matrix[:, 2] + data_in_matrix[:, 3])
    seq16 = np.conj(seq16qam).tolist()
    seq16.reverse()

    X = np.zeros((1, int(n_bits/2)), dtype=complex)
    X[0] = np.append(seq16qam, np.asarray(seq16, dtype=complex))

    xn = np.zeros((1, int(N)), dtype=complex)

    for n in range(int(N)):
        for k in range(int(N)):
            xn[0, n] = xn[0, n] + 1 / np.sqrt(N) * X[0, k] * np.exp(i * 2 * PI * n * k / N)

    for ik in range(len(sigmas)):

        variance = sigmas[ik]

        noise = np.sqrt(variance) * np.random.randn(1, int(N)) + 1j * np.sqrt(variance) * np.random.randn(1, int(N))

        rn = xn + noise
        Y = np.zeros((1, int(K)), dtype=complex)

        for k in range(0, int(K)):
            for n in range(0, int(N)):
                Y[0, k] = Y[0, k] + 1/np.sqrt(N)*rn[0, n]*np.exp(-1j*2*np.pi*k*n/N)
        # print(np.shape(Y))
        plt.scatter(Y.real, Y.imag)
        plt.scatter(X.real, X.imag, color='red', marker='+')
        plt.title(f'Sinal com ruído de variância {variance}')
        plt.show()

        Z = np.zeros(np.shape(Y), dtype=complex)

        for k in range(len(Y[0])):
            if np.real(Y[0, k]) > 0:
                if np.real(Y[0, k]) > 2:
                    Z[0, k] = 3
                else:
                    Z[0, k] = 1
            else:
                if np.real(Y[0, k]) < -2:
                    Z[0, k] = -3
                else:
                    Z[0, k] = -1

            if np.imag(Y[0, k]) > 0:
                if np.imag(Y[0, k]) > 2:
                    Z[0, k] = Z[0, k] + (i * 3)
                else:
                    Z[0, k] = Z[0, k] + i
            else:
                if np.imag(Y[0, k]) < -2:
                    Z[0, k] = Z[0, k] - (i * 3)
                else:
                    Z[0, k] = Z[0, k] - i

        print(np.shape(Z))
        error = len(np.nonzero(Z[0, 1:int(K)]-X[0, 1: int(K)])[0])
        print(f'Para variância de , {variance}, houve , {error}, símbolos errados.')
