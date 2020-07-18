# coding: utf-8
__author__ = "David Morais"
__copyright__ = "Copyright 2020, Projeto 02, Comunicacoes Moveis"
__credits__ = ["David Morais"]
__maintainer__ = "David Morais"
__email__ = "moraisdavid8@gmail.com"
__status__ = "Development"
__version__ = "0.0.1-SNAPSHOT"

# Builtin imports
import sys
import time

# Project imports
from entregas.QAM16 import QAM16

# PARAMETERS
modulation_params = {"nbits": 100, "ts": 50, "ts_carrier": 2}
demodulation_params = {"nbits": 24000, "ts": 500, "ts_carrier": 2}


if __name__ == '__main__':

    print('\n_____THIS IS A PROGRAM TO SIMULATE OFDM TRANSMISSION_____')
    print('SELECT ONE OF THE OPTIONS BELOW TO VERIFY')
    option = int(input('(1) 16QAM - (2) BPSK: '))

    while True:

        if option == 1:

            print('16QAM SIMULATION SELECTED, INITIATING...\n')
            time.sleep(2)

            mod = QAM16(modulation_params["nbits"], modulation_params["ts"], modulation_params["ts_carrier"])
            mod.modulate()

            dmod = QAM16(demodulation_params["nbits"], demodulation_params["ts"], demodulation_params["ts_carrier"])
            dmod.demodulate()

            rerun = input('\nFINALIZED. DO YOU WANT TO RUN ANOTHER SIMULATION? (y ou n): ')

            if rerun.lower() == 'y':

                print('SELECT ONE OF THE OPTIONS BELOW TO VERIFY')
                option = int(input('(1) 16QAM - (2) BPSK: '))

            else:

                print("THANKS. TERMINATING THE PROGRAM...")
                time.sleep(1)
                sys.exit(0)

        elif option == 2:

            print('\nBPSK SIMULATION SELECTED, INITIATING...')
            time.sleep(2)

            bpsk_mod = QAM16(modulation_params["nbits"], modulation_params["ts"], modulation_params["ts_carrier"])
            bpsk_mod.modulate()

            rerun = input('\nFINALIZED. DO YOU WANT TO RUN ANOTHER SIMULATION? (y ou n): ')

            if rerun.lower() == 'y':

                print('SELECT ONE OF THE OPTIONS BELOW TO VERIFY')
                option = int(input('(1) 16QAM - (2) BPSK: '))

            else:

                print("THANKS. TERMINATING THE PROGRAM...")
                time.sleep(1)
                sys.exit(0)

        else:

            print('WRONG OPTION ENTERED. TRY AGAIN')
            option = int(input('(1) 16QAM - (2) BPSK: '))

            if option not in [1, 2]:

                print('TWO CONSECUTIVE WRONG ENTRIES. TERMINATING...')
                time.sleep(0.5)
                sys.exit(1)
