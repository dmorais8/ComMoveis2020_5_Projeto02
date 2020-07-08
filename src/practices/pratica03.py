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
sigmas = np.array([0, 0.1, 1])
i = 1j

if __name__ == '__main__':

    data_in = np.random.rand(1, n_bits)
    data_in = np.sign(data_in - .5)
    data_in_matrix = data_in.reshape((n_bits//4, 4))

    seq16qam = 2 * data_in_matrix[:, 0] + data_in_matrix[:, 1] + i * (2 * data_in_matrix[:, 2] + data_in_matrix[:, 3])
    seq16qam_conj_reverse = np.conj(seq16qam).tolist()
    seq16qam_conj_reverse.reverse()

    X = np.append(seq16qam, np.asarray(seq16qam_conj_reverse, dtype=complex))

    xn = np.zeros((1, int(N)), dtype=complex)
    for n in range(int(N)):
        for k in range(int(N)):
            xn[0, n] = xn[0, n] + 1 / np.sqrt(N) * X[k] * np.exp(i * 2 * PI * n * k / N)

    # print(len(sigmas))
    for ik in range(len(sigmas)):

        variance = sigmas[ik]

        noise = np.sqrt(variance) * np.random.rand(1, int(N)) + i * np.sqrt(variance) * np.random.rand(1, int(N))

        rn = xn + noise

        Y = np.zeros((1, int(K)), dtype=complex)

        for k in range(int(K)):
            for n in range(int(N)):
                Y[0, k] = Y[0, k] + 1 / np.sqrt(N) * rn[n] * np.exp(-i * 2 * PI * k * n / N)

        plt.scatter(Y, np.real(seq16qam), np.imag(seq16qam), color='red', marker='+')
        plt.title(f'Sinal com ruído de variância {variance}')

        Z = np.zeros(np.shape(Y))

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

        error =

    print(len(Y[0]))