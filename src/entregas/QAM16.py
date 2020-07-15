import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt


class QAM16:

    i = 1j
    PI = np.pi
    bit_energy = 1                                   # Energia de bit
    ebnodb_array = np.arange(0, 15)               # EbNo em dB
    ebno_linear_array = bit_energy * (10. ** (-ebnodb_array / 10))  # Potência do ruído

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

        seq16qam = 2 * data_in_matrix[:, 0] + data_in_matrix[:, 1] + QAM16.i * (2 * data_in_matrix[:, 2] +
                                                                                data_in_matrix[:, 3])
        seq16 = np.conj(seq16qam).tolist()
        seq16.reverse()

        X = np.zeros((1, int(self.num_bits/2)), dtype=complex)
        X[0] = np.append(seq16qam, np.asarray(seq16, dtype=complex))

        return X

    def modulation(self, X):

        xn = np.zeros((1, int(self.N)), dtype=complex)
        for n in range(int(self.N)):
            for k in range(int(self.N)):
                xn[0, n] = xn[0, n] + 1 / np.sqrt(self.N) * X[0, k] * np.exp(QAM16.i * 2 * QAM16.PI * n * k / self.N)

        xt = np.zeros((1, int(self.tsimb + 1)), dtype=complex)
        for t in range(int(self.tsimb + 1)):
            xt = ifft(xn, axis=0)

        return {"x_discrete": xn, "x_analog": xt}

    def modulate(self):

        X = self.gen_constelation_16qam()
        signals = self.modulation(X)

        xt = signals['x_analog']
        xn = signals['x_discrete']

        plt.plot(np.abs(xn[0]), label="x(t)")
        markerline, stemlines, baseline = plt.stem(np.abs(xt[0]), use_line_collection=True, linefmt='red',
                                                   markerfmt='bo', label="x_n")
        markerline.set_markerfacecolor('none')
        plt.title('Sinais OFDM')
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
                           np.exp(QAM16.i * 2 * QAM16.PI * n * k / self.N)

        for ik in range(len(QAM16.ebnodb_array)):

            noise = np.sqrt(QAM16.ebno_linear_array[ik]) * np.random.randn(1, int(self.N)) + \
                    1j * np.sqrt(QAM16.ebno_linear_array[ik]) * np.random.randn(1, int(self.N))

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
                        Z[0, k] = Z[0, k] + (QAM16.i * 3)
                    else:
                        Z[0, k] = Z[0, k] + QAM16.i
                else:
                    if np.imag(Y[0, k]) < -2:
                        Z[0, k] = Z[0, k] - (QAM16.i * 3)
                    else:
                        Z[0, k] = Z[0, k] - QAM16.i

            zarrays.append(Z)

        return {
            'yarrays': yarrays,
            'zarrays': zarrays
        }

    def demodulate(self):

        X = self.gen_constelation_16qam()
        signals = self.demodulation(X)

        fig, axs = plt.subplots(3, 3)

        for i in range(3):
            for j in range(3):

                if i == 0 and j == 0:
                    ebn0 = 0
                if i == 0 and j == 1:
                    ebn0 = 1
                if i == 0 and j == 2:
                    ebn0 = 2
                if i == 1 and j == 0:
                    ebn0 = 3
                if i == 1 and j == 1:
                    ebn0 = 4
                if i == 1 and j == 2:
                    ebn0 = 5
                if i == 2 and j == 0:
                    ebn0 = 6
                if i == 2 and j == 1:
                    ebn0 = 7
                if i == 2 and j == 2:
                    ebn0 = 8

                axs[i, j].scatter(signals['yarrays'][ebn0].real, signals['yarrays'][ebn0].imag, marker='.')
                axs[i, j].scatter(X.real, X.imag, color='red', marker='+')
                axs[i, j].set_title(f'EbN0 {QAM16.ebnodb_array[ebn0]}')


            # axs[0, 1].scatter(signals['yarrays'][1].real, signals['yarrays'][1].imag, marker='.')
            # axs[0, 1].scatter(X.real, X.imag, color='red', marker='+')
            # axs[0, 1].set_title(f'EbN0 {QAM16.ebnodb_array[1]}')
            # axs[0, 2].scatter(signals['yarrays'][2].real, signals['yarrays'][2].imag, marker='.')
            # axs[0, 2].scatter(X.real, X.imag, color='red', marker='+')
            # axs[0, 2].set_title(f'EbN0 {QAM16.ebnodb_array[2]}')
            #
            # axs[1, 0].scatter(signals['yarrays'][3].real, signals['yarrays'][3].imag, marker='.')
            # axs[1, 0].scatter(X.real, X.imag, color='red', marker='+')
            # axs[1, 0].set_title(f'EbN0 {QAM16.ebnodb_array[3]}')
            # axs[1, 1].scatter(signals['yarrays'][4].real, signals['yarrays'][4].imag, marker='.')
            # axs[1, 1].scatter(X.real, X.imag, color='red', marker='+')
            # axs[1, 1].set_title(f'EbN0 {QAM16.ebnodb_array[4]}')
            # axs[1, 2].scatter(signals['yarrays'][5].real, signals['yarrays'][5].imag, marker='.')
            # axs[1, 2].scatter(X.real, X.imag, color='red', marker='+')
            # axs[1, 2].set_title(f'EbN0 {QAM16.ebnodb_array[5]}')

        # for ebn0 in range(len(QAM16.ebnodb_array)):
        #
        #     # plt.figure(ebn0)
        #     # plt.scatter(signals['yarrays'][ebn0].real, signals['yarrays'][ebn0].imag, marker='.')
        #     # plt.scatter(X.real, X.imag, color='red', marker='+')
        #     # plt.title(f'Sinal com Eb/N0 de variância {ebn0}')
        #
        #     Z = signals['zarrays'][ebn0]
        #     nonzeroarray = np.nonzero(Z[0, 1:int(self.K)] - X[0, 1:int(self.K)])[0]
        #     sigma = nonzeroarray.sum() / self.num_bits
        #     print(f'Para um EbNo de {ebn0}dB, a variancia eh de {sigma:.2f}')
        #

        plt.show()
