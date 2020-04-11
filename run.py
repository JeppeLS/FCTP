import extFCTP
import numpy as np
import pandas as pd

dir = './'
files = ['N207.FCTP', 'N307.FCTP', 'N507.FCTP', 'N1007.FCTP', 'N2007.FCTP', 'N3304.FCTP', 'N3507.FCTP', 'N3704.FCTP']

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
        df = pd.DataFrame({'K-values': range(10)})
        for data_file in files[2:5]:
            fctp = extFCTP.extFCTP(data_file=data_file, ini_file=ini_file)
            res = []
            for k in range(10):
                extFCTP.FCTP.param.set(29, k) # Set k-step value
                fctp.solve()
                res.append(fctp.solution.tot_cost)
            df[data_file] = res.copy()
        df.to_latex(buf=(dir + ini_file + '_k_step_test'))

def test_instances():
    ini_files = ['ILS.ini', 'ILSReset.ini', 'VanillaILS.ini']
    df = pd.DataFrame({'File Names': files})
    for ini_file in ini_files:
        res = []
        for data_file in files:
            fctp = extFCTP.extFCTP(data_file=data_file, ini_file=ini_file)
            fctp.solve()
            res.append(fctp.solution.tot_cost)
        df[ini_file] = res
    df.to_latex(buf=(dir + 'results'))

def test_instances_v2():
    ini_files = ['ILSBlockMove.ini', 'ILS.ini']
    df = pd.DataFrame({'File Names': files[4:5]})
    for ini_file in ini_files:
        res = []
        for data_file in files[4:5]:
            fctp = extFCTP.extFCTP(data_file=data_file, ini_file=ini_file)
            fctp.solve()
            res.append(fctp.solution.tot_cost)
        df[ini_file] = res
    df.to_latex(buf=(dir + 'res_test'))



if __name__ == "__main__":
    test_instances_v2()