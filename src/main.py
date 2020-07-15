# coding: utf-8
__author__ = "David Morais"
__copyright__ = "Copyright 2020, Projeto 02, Comunicacoes Moveis"
__credits__ = ["David Morais"]
__maintainer__ = "David Morais"
__email__ = "moraisdavid8@gmail.com"
__status__ = "Development"
__version__ = "0.0.1-SNAPSHOT"

# Project imports
from entregas.QAM16 import QAM16


if __name__ == '__main__':

    Ts = 2

    # Modulacao
    n_bits_mod = 100  # Número de bits
    T_mod = 50  # Tempo de símbolo

    mod = QAM16(n_bits_mod, T_mod, Ts)
    mod.modulate()

    # Demodulacao
    n_bits_dmod = 1000  # Número de bits
    T_dmod = 500  # Tempo de símbolo

    dmod = QAM16(n_bits_dmod, T_dmod, Ts)
    dmod.demodulate()
