import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from scipy.special import erfc
from .QAM16 import QAM16

# CONSTANTS
i = 1j
PI = np.pi


def theoretical_ber_bpsk():

    return 2. * (1 - (1 / np.sqrt(BPSK.mod_bits))) * erfc(np.sqrt(3 * BPSK.ebn0_linear_array /
                                                                  (2 * (BPSK.mod_bits - 1))))


# noinspection DuplicatedCode
class BPSK:

    bit_energy = 1  # Energia de bit
    ebn0db_array = np.arange(0, 15)  # EbNo em dB
    ebn0_linear_array = 10. ** (ebn0db_array / 10)  # ebn0db_array em escala linear
    n0_power_array = bit_energy * (10. ** (-ebn0db_array / 10))  # Potência do ruído
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
        seqbpsk_conj = np.conj(seqbpsk).tolist()
        seqbpsk_conj.reverse()

        X = np.zeros((1, int(self.num_bits/2)), dtype=complex)
        X[0] = np.append(seqbpsk, np.asarray(seqbpsk_conj, dtype=complex))

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
        plt.title('Sinais OFDM')
        plt.legend(loc="upper left")
        plt.xlabel('Tempo')
        plt.show(block=False)
