import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from scipy.special import erfc
from .QAM16 import QAM16

# CONSTANTS
i = 1j
PI = np.pi


def theoretical_ber_bpsk():

    return 1 / 2. * erfc(np.sqrt(BPSK.ebn0_linear_array))


# noinspection DuplicatedCode
class BPSK:

    bit_energy = 1  # Energia de bit
    ebn0db_array = np.arange(0, 16)  # EbNo em dB
    ebn0_linear_array = 10. ** (ebn0db_array / 10)  # ebn0db_array em escala linear
    n0_power_array = bit_energy * 10. ** (-ebn0db_array / 10)  # Potência do ruído
    mod_simbols = 2  # Número de simbolos no 16-QAM
    mod_bits = np.log2(mod_simbols)  # Número de bits por simbolo no 16-QAM

    def __init__(self, num_bits, tsimb, tsimb_single_carr):
        self.tsimb_single_carr = tsimb_single_carr
        self.tsimb = tsimb
        self.num_bits = num_bits
        self.K = self.tsimb / self.tsimb_single_carr
        self.N = 2 * self.K

    def gen_constelation_bpsk(self):

        data_in = np.random.rand(1, self.num_bits)
        data_in = np.sign(data_in - .5)
        data_in_matrix = data_in.reshape((self.num_bits // 1, 1))

        seqbpsk = data_in_matrix[:, 0]
        seqbpsk_trsp = np.transpose(seqbpsk)
        seqbpsk_conj = np.conj(seqbpsk_trsp).tolist()
        seqbpsk_conj.reverse()

        X = np.zeros((1, int(2 * self.num_bits)), dtype=complex)
        X[0] = np.append(seqbpsk_trsp, np.asarray(seqbpsk_conj, dtype=complex))

        return X

    def modulate(self):

        X = self.gen_constelation_bpsk()
        signals = QAM16.gen_signals(X, self.N, self.tsimb)

        xt = signals['x_analog']
        xn = signals['x_discrete']

        plt.plot(np.abs(xn[0]), label="x(t)")
        markerline, stemlines, baseline = plt.stem(np.abs(xt[0]), use_line_collection=True, linefmt='red',
                                                   markerfmt='bo', label="x_n")
        markerline.set_markerfacecolor('none')
        plt.title('Sinais OFDM com BPSK')
        plt.legend(loc="upper left")
        plt.xlabel('Tempo')
        plt.show(block=False)

    def demodulate(self, X):

        yarrays = []
        zarrays = []

        xn = np.zeros((1, int(self.N)), dtype=complex)

        for n in range(int(self.N)):
            for k in range(int(self.N)):
                xn[0, n] = xn[0, n] + 1 / np.sqrt(self.N) * X[0, k] * \
                           np.exp(i * 2 * PI * n * k / self.N)

        for ik in range(len(BPSK.ebn0db_array)):

            noise = np.sqrt(BPSK.n0_power_array[ik]) * np.random.randn(1, int(self.N)) + \
                    i * np.sqrt(BPSK.n0_power_array[ik]) * np.random.randn(1, int(self.N))

            rn = xn + noise

            Y = np.zeros((1, int(self.K)), dtype=complex)

            for k in range(0, int(self.K)):

                Y = fft(rn/22)

            yarrays.append(Y)

            Z = np.zeros(np.shape(Y), dtype=complex)

            for k in range(len(Y[0])):

                if np.real(Y[0, k]) > 0:

                    Z[0, k] = 1

                else:

                    Z[0, k] = -1

            zarrays.append(Z)

        return {
            'yarrays': yarrays,
            'zarrays': zarrays
        }

    def simulate(self):

        X = self.gen_constelation_bpsk()
        signals = self.demodulate(X)

        bpsk_ber = []

        for ebn0 in range(len(BPSK.ebn0db_array)):

            plt.figure(ebn0)
            plt.scatter(signals['yarrays'][ebn0].real, signals['yarrays'][ebn0].imag, marker='.')
            plt.scatter(X.real, X.imag, color='red', marker='+')
            plt.title(f'Sinal OFDM/BPSK com Eb/N0 de {ebn0}dB')

            Z = signals['zarrays'][ebn0]
            nonzeroarray = np.nonzero(Z[0, 1:int(self.K)] - X[0, 1:int(self.K)])[0]
            sigma = (sum(nonzeroarray) / self.num_bits) ** 2
            error = len(nonzeroarray)
            bpsk_ber.append(error / self.num_bits)
            print(f'Para um Eb/N0 de {ebn0}dB, a variancia eh de {sigma}')

        plt.show(block=False)

        plt.figure(300)
        plt.semilogy(BPSK.ebn0db_array, theoretical_ber_bpsk(),
                     label='theoretical')
        plt.semilogy(bpsk_ber, 'r-', label='simulated')
        plt.grid(color='g', linestyle='-', linewidth=0.2)
        plt.title('BER vs Eb/N0 for BPSK')
        plt.ylabel('BER')
        plt.xlabel('Eb/N0')
        plt.legend(loc='upper right')

        plt.show(block=False)
