import extFCTP
import numpy as np
import pandas as pd

dir = './Results/'

## Testing iterations needed:
def itertest(ini_files, files):
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


def kvaluetest(ini_files, files, min_k, max_k):
    for ini_file in ini_files:
        df = pd.DataFrame({'K-values': range(min_k, max_k+1)})
        for data_file in files:
            fctp = extFCTP.extFCTP(data_file=data_file, ini_file=ini_file)
            res = []
            for k in range(min_k, max_k + 1):
                print('K = ', k)
                extFCTP.FCTP.param.set(29, k) # Set k-step value
                extFCTP.FCTP.param.set(4, 100) # Set Max Iter
                extFCTP.FCTP.param.set(12, 10) # Set Runs
                fctp.solve()
                res.append(fctp.solution.tot_cost)
            df[data_file] = res.copy()
        df.to_latex(buf=(dir + ini_file + '_k_step_test'))

def test_instances(ini_files, files):
    df = pd.DataFrame({'File Names': files})
    for ini_file in ini_files:
        res = []
        for data_file in files:
            fctp = extFCTP.extFCTP(data_file=data_file, ini_file=ini_file)
            param = extFCTP.FCTP.param
            param.set(param.kstep, 4)
            param.set(param.max_iter, 500)
            param.set(param.max_before_reset, 20)
            param.set(param.weight_func, 'something')
            param.set(param.ils_type, param.ils_standard)
            fctp.solve()
            res.append(fctp.solution.tot_cost)
        df[ini_file] = res
    df.to_latex(buf=(dir + 'resultsv2'))

def test_instances_v2(ini_files, files):
    df = pd.DataFrame({'File Names': files})
    for ini_file in ini_files:
        res = []
        for _ in range(20):
            for data_file in files:
                fctp = extFCTP.extFCTP(data_file=data_file, ini_file=ini_file)
                extFCTP.FCTP.param.set(4, 100)
                extFCTP.FCTP.param.set(12, 1)
                fctp.solve()
                res.append(fctp.solution.tot_cost)
        df[ini_file] = np.mean(res)
    df.to_latex(buf=(dir + 'res_test'))

if __name__ == "__main__":
    files = ['N207.FCTP', 'N307.FCTP', 'N507.FCTP','N1007.FCTP', 'N2007.FCTP', 'N3304.FCTP', 'N3507.FCTP',
             'N3704.FCTP']
    ini_files = ['ILSLinear.ini', 'ILSLinearReset.ini']
    kvaluetest(ini_files, files[2:4], 1, 10)