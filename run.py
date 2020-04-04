import extFCTP
import numpy as np
import os
import pandas as pd

dir = './'
files = []
for filename in os.listdir(dir):
    if filename.endswith('.FCTP'):
        files.append(str(filename))

## Testing iterations needed:
def itertest():
    ini_files = ['ILS.ini', 'ILSReset.ini', 'VanillaILS.ini']
    df = pd.DataFrame({'File Names': files})
    for ini_file in ini_files:
        res = []
        for data_file in files:
            fctp = extFCTP.extFCTP(data_file=data_file, ini_file=ini_file)
            fctp.solve()
            best_sol_iter = np.argmin(fctp.history)
            res.append(best_sol_iter)
        df[ini_file] = res.copy()
    df.to_latex(buf=(dir + 'iter_test'))

def kvaluetest():
    ini_files = ['ILS.ini', 'ILSReset.ini']
    for ini_file in ini_files:
        df = pd.DataFrame({'K-values': range(1)})
        for data_file in files[0:10]:
            fctp = extFCTP.extFCTP(data_file=data_file, ini_file=ini_file)
            res = []
            for k in range(1):
                extFCTP.FCTP.param.set(29, k) # Set k-step value
                fctp.solve()
                res.append(fctp.solution.tot_cost)
            df[data_file] = res.copy()
        df.to_latex(buf=(dir + ini_file + '_k_step_test'))
kvaluetest()

def test_instances():
    df = pd.DataFrame({'File Names': files})
    for ini_file in ini_files:
        res = []
        for data_file in files:
            fctp = extFCTP.extFCTP(data_file=data_file, ini_file=ini_file)
            fctp.solve()
            res.append(fctp.solution.tot_cost)
        df[ini_file] = res






