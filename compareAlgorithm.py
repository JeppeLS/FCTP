import os

from extFCTP import extFCTP
from sys import argv
from getopt import getopt
from tempfile import NamedTemporaryFile
from FCTP.convert import convert
import numpy as np

dir = './'
for filename in os.listdir(dir):
    print(filename)
    if filename.endswith('.FCTP'):
        data_file = filename
        os.system('python3 tryFCTP.py ' + data_file + ' -i ILS.ini')
        os.system('python3 tryFCTP.py ' + data_file + ' -i VanillaILS.ini')