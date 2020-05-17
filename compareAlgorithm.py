import os

from extFCTP import extFCTP
from sys import argv
from getopt import getopt
from tempfile import NamedTemporaryFile
from FCTP.convert import convert
import numpy as np

files = ['N3401.FCTP', 'N3501.FCTP', 'N3704.FCTP']
for filename in files:
    data_file = filename
    os.system('python3 tryFCTP.py ' + data_file + ' -i ILSHist.ini')
    os.system('python3 tryFCTP.py ' + data_file + ' -i ILSStandard.ini')