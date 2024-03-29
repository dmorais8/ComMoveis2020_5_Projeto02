import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from scipy.special import erfc

i = 1j
PI = np.pi


def theoretical_ber_16qam():

    error_probability = 2. * (1 - (1 / np.sqrt(16))) * erfc(np.sqrt(3 * QAM16.ebn0_linear_array /
                                                                    (2 * (16 - 1))))

    return 1 / 2. * error_probability


class QAM16:

    bit_energy = 1                                   # Energia de bit
    ebn0db_array = np.arange(0, 15)               # EbNo em dB
    ebn0_linear_array = 10. ** (ebn0db_array / 10)       # ebn0db_array em escala linear
    n0_power_array = bit_energy * (10. ** (-ebn0db_array / 10))  # Potência do ruído
    mod_simbols = 16    # Número de simbolos no 16-QAM
    mod_bits = np.log2(mod_simbols)    # Número de bits por simbolo no 16-QAM

    def __init__(self, num_bits, tsimb, tsimb_single_carr):
        self.num_bits = num_bits
        self.tsimb = tsimb
        self.tsimb_single_carr = tsimb_single_carr
        self.K = self.tsimb / self.tsimb_single_carr
        self.N = 2 * self.K

    def gen_constelation_16qam(self):

        data_in = np.random.rand(1, self.num_bits)
        data_in = np.sign(data_in - .5)
        data_in_matrix = data_in.reshape((self.num_bits // 4, 4))

        seq16qam = 2 * data_in_matrix[:, 0] + data_in_matrix[:, 1] + i * (2 * data_in_matrix[:, 2] +
                                                                                data_in_matrix[:, 3])
        seq16 = np.conj(seq16qam).tolist()
        seq16.reverse()

        X = np.zeros((1, int(self.num_bits/2)), dtype=complex)
        X[0] = np.append(seq16qam, np.asarray(seq16, dtype=complex))

        return X

    @staticmethod
    def gen_signals(X, N, tsimb):

        xn = np.zeros((1, int(N)), dtype=complex)
        for n in range(int(N)):
            for k in range(int(N)):
                xn[0, n] = xn[0, n] + 1 / np.sqrt(N) * X[0, k] * np.exp(i * 2 * PI * n * k / N)

        xt = np.zeros((1, int(tsimb + 1)), dtype=complex)
        for t in range(int(tsimb + 1)):
            xt = ifft(xn, axis=0)

        return {"x_discrete": xn, "x_analog": xt}

    def modulate(self):

        X = self.gen_constelation_16qam()
        signals = self.gen_signals(X, self.N, self.tsimb)

        xt = signals['x_analog']
        xn = signals['x_discrete']

        plt.figure(200)
        plt.plot(np.abs(xn[0]), label="x(t)")
        markerline, stemlines, baseline = plt.stem(np.abs(xt[0]), use_line_collection=True, linefmt='red',
                                                   markerfmt='bo', label="x_n")
        markerline.set_markerfacecolor('none')
        plt.title('Sinais OFDM para 16-QAM')
        plt.legend(loc="upper left")
        plt.xlabel('Tempo')
        plt.show(block=False)

    def demodulation(self, X):

        yarrays = []
        zarrays = []

        xn = np.zeros((1, int(self.N)), dtype=complex)

        for n in range(int(self.N)):
            for k in range(int(self.N)):
                xn[0, n] = xn[0, n] + 1 / np.sqrt(self.N) * X[0, k] * \
                           np.exp(i * 2 * PI * n * k / self.N)

        for ik in range(len(QAM16.ebn0db_array)):

            noise = np.sqrt(QAM16.n0_power_array[ik]) * np.random.randn(1, int(self.N)) + \
                    1j * np.sqrt(QAM16.n0_power_array[ik]) * np.random.randn(1, int(self.N))

            rn = xn + noise

            Y = np.zeros((1, int(self.K)), dtype=complex)

            for k in range(0, int(self.K)):

                Y = fft(rn/22)

            yarrays.append(Y)

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

            zarrays.append(Z)

        return {
            'yarrays': yarrays,
            'zarrays': zarrays
        }

    def demodulate(self):

        X = self.gen_constelation_16qam()
        signals = self.demodulation(X)

        qam16_ber = []

        for ebn0 in range(len(QAM16.ebn0db_array)):

            plt.figure(ebn0)
            plt.scatter(signals['yarrays'][ebn0].real, signals['yarrays'][ebn0].imag, marker='.')
            plt.scatter(X.real, X.imag, color='red', marker='+')
            plt.title(f'Sinal OFDM/16-QAM com Eb/N0 de {ebn0}dB')

            Z = signals['zarrays'][ebn0]
            nonzeroarray = np.nonzero(Z[0, 1:int(self.K)] - X[0, 1:int(self.K)])[0]
            sigma = (sum(nonzeroarray) / self.num_bits) ** 2
            error = len(nonzeroarray)
            qam16_ber.append(4 * (error / self.num_bits))
            print(f'Para um Eb/N0 de {ebn0}dB, a variancia eh de {sigma:.5f}')

        plt.show(block=False)

        plt.figure(300)
        plt.semilogy(QAM16.ebn0db_array, theoretical_ber_16qam(),
                     label='theoretical')
        plt.semilogy(QAM16.ebn0db_array, qam16_ber, 'r-', label='simulated')
        plt.grid(color='g', linestyle='-', linewidth=0.1)
        plt.title('BER vs Eb/N0 for 16-QAM')
        plt.ylabel('BER')
        plt.xlabel('Eb/N0')
        plt.legend(loc='upper right')

        plt.show(block=False)

