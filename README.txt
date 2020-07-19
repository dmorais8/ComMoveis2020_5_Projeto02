SOBRE

Codigos do Projeto 02 da disciplina de Comunicacoes Moveis no semestre 2020.5
Multiplexação OFDM (ortogonalidade, transmissão e recepção, desempenho em canal AWGN)

Objetivos
As metas desse projeto são ajudar o usuário a:

    * Entender a modelagem da multiplexação OFDM;
    * Entender o processo de ortogalização entre subportadoras OFDM;
    * Entender a modelagem da demultiplexação OFDM;
    * Demonstrar o processo de demultiplexação OFDM em canais AWGN.

ESTRUTURA DO PROJETO

├── README.txt
├── requirements.txt
└── src
    ├── __init__.py
    ├── entregas
    │   ├── BPSK.py
    │   ├── QAM16.py
    │   └── __init__.py
    ├── main.py
    ├── modules
    │   ├── Functions.py
    │   └── __init__.py
    └── practices
        ├── __init__.py
        ├── pratica01.py
        ├── pratica02.py
        └── pratica03.py

REQUISITOS:

    - Windows
        * Python 3.6 ou mais recente
    - Linux (debian like)
        * Python 3.6 ou mais recente
        * python3-venv

PREPARANDO O AMBIENTE:

Windows:
No prompt de comando.

cd projeto02
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt

Linux(Ubuntu):
No terminal.

cd projeto02
sudo apt-get install python3-venv python3-tk
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

EXECUTANDO O PROJETO

Dentro do diretorio raiz do projeto, faca:

Windows:
    - python src\main.py

Linux:
    - python src/main.py