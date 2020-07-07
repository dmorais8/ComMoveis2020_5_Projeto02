# coding: utf-8
__author__ = "David Morais"
__copyright__ = "Copyright 2020, Projeto 01, Comunicacoes Moveis"
__credits__ = ["David Morais"]
__maintainer__ = "David Morais"
__email__ = "moraisdavid8@gmail.com"
__status__ = "Development"
__version__ = "0.0.1-SNAPSHOT"

# External imports
import numpy as np
from matplotlib import pyplot as plt

#Project imports
from Functions import Pessoa


if __name__ == "__main__":

    pessoa = Pessoa()
    pessoa.set_nome("David")
    print(pessoa.get_nome())