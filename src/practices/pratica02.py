# coding: utf-8
__author__ = "David Morais"

# Lib imports
import numpy as np
from matplotlib import pyplot as plt

# CONSTANTS
n_bits = 100    # Número de bits
T = 50          # Tempo de símbolo
Ts = 2          # Tempo de símbolo em portadora única
K = T/Ts        # Número de subportadoras independentes
N = 2*K         # N pontos da IDFT
PI = np.pi
i = 1j

if __name__ == '__main__':

    data_in = np.random.rand(1, n_bits)
    data_in = np.sign(data_in - .5)
    data_in_matrix = data_in.reshape((n_bits//4, 4))

    seq16qam = 2 * data_in_matrix[:, 0]+data_in_matrix[:, 1] + i * (2 * data_in_matrix[:, 2] + data_in_matrix[:, 3])
    seq16qam_conf_reverse = np.conj(seq16qam).tolist()
    seq16qam_conf_reverse.reverse()

    X = np.append(seq16qam, np.asarray(seq16qam_conf_reverse, dtype=complex))

    xn = np.zeros((1, int(N)), dtype=complex)
    for n in range(int(N)):
        for k in range(int(N)):
            xn[0, n] = xn[0, n] + 1 / np.sqrt(N) * X[k] * np.exp(i * 2 * PI * n * k / N)

    xt = np.zeros((1, int(T)), dtype=complex)
    # xt = xn
    for t in range(int(T)):
        for k in range(int(N)):
            xt[0, t] = xt[0, t] + 1 / np.sqrt(N) * X[k] * np.exp(i * 2 * PI * k * t / T)

    plt.plot(np.abs(xn[0]), label="x(t)")
    markerline, stemlines, baseline = plt.stem(np.abs(xt[0]), use_line_collection=True, linefmt='red',
                                               markerfmt='go', label="x_n")
    markerline.set_markerfacecolor('none')
    plt.title('Sinais OFDM')
    plt.legend(loc="upper left")
    plt.xlabel('Tempo')
    plt.show()

