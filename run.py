import time
import timeit

import extFCTP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import rc

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

def rvaluetest(ini_files, files, min_r, max_r):
    for ini_file in ini_files:
        df = pd.DataFrame({'R-values': range(min_r, max_r+1,5)})
        for data_file in files:
            fctp = extFCTP.extFCTP(data_file=data_file, ini_file=ini_file)
            res = []
            for r in range(min_r, max_r + 1,5):
                print('R = ', r)
                param = extFCTP.FCTP.param
                param.set(29, 5) # Set k-step value
                param.set(4, 100) # Set Max Iter
                param.set(12, 10) # Set Runs
                param.set(30, r)  # Set max_before_reset
                param.set(32, True)  # Set reset
                param.set(28, 1)
                fctp.solve()
                res.append(fctp.solution.tot_cost)
            df[data_file] = res.copy()
        df.to_latex(buf=(dir + ini_file + '_r_test'))

def test_weight_func(ini_files, files):
    df = pd.DataFrame({'File Names': files})
    for ini_file in ini_files:
        res = []
        for data_file in files:
            fctp = extFCTP.extFCTP(data_file=data_file, ini_file=ini_file)
            param = extFCTP.FCTP.param
            param.set(param.kstep, 5)
            param.set(param.max_iter, 500)
            param.set(param.max_before_reset, 20)
            param.set(param.ils_type, param.ils_kstep)
            fctp.solve()
            res.append(fctp.solution.tot_cost)
        df[ini_file] = res
    df.to_latex(buf=(dir + 'test_weight_func'))


def plot(fctp, ini_file, data_file):
    if not fctp.all_hist is None:
        font = {'family': 'serif',
                'serif': ['computer modern roman'],
                'size': 16}
        rc('text', usetex=True)
        rc('font', **font)
        max_iter = max([len(h) for h in fctp.all_hist])
        hist_dat = np.zeros(max_iter, dtype=float)
        for h in fctp.all_hist: hist_dat[:len(h)] += np.array(h)
        nruns = len(fctp.all_hist)
        count = [len([h for h in fctp.all_hist if len(h) > i]) for i in range(max_iter)]
        hist_dat /= np.array(count)
        if ini_file == 'ILSHist.ini':
            heuristic = 'Standard ILS'
        else:
            heuristic = 'Adaptive ILS'
        plt.title("Search history in " + data_file + ' using ' + heuristic)
        plt.plot(hist_dat, marker='o', color='blue', markersize=2)
        plt.tight_layout()
        path = os.getcwd() + '/' + str(data_file) + '_' + str(ini_file) + '.pdf'
        print(path)
        plt.savefig(fname=path)
        plt.close()


def test_instances(ini_files, files, num_runs):
    df = pd.DataFrame({'File Names': files})
    for ini_file in ini_files:
        res = []
        for data_file in files:
            mean = []
            for run in range(num_runs):
                fctp = extFCTP.extFCTP(data_file=data_file, ini_file=ini_file)
                param = extFCTP.FCTP.param
                param.set(param.max_iter, 500)
                param.set(param.num_runs, 1)
                param.set(param.max_before_diversify, 10)
                param.set(param.iter_to_diversify, 5)
                fctp.solve()
                mean.append(fctp.solution.tot_cost)
                print(data_file + ': ' + str(mean))
                plot(fctp, ini_file, data_file)
            res.append(np.mean(mean))
        df[ini_file] = res
    df.to_latex(buf=(dir + 'results_1_run_weighting_num_sol_500_iter'))

def test_max_before_diversify(ini_files, files, search_range, num_runs):
    df = pd.DataFrame({'Iterations with no improvement before diversification producedure is initiated': search_range})
    for ini_file in ini_files:
        for data_file in files:
            res = []
            for diversify in search_range:
                mean = []
                for run in range(num_runs):
                    fctp = extFCTP.extFCTP(data_file=data_file, ini_file=ini_file)
                    param = extFCTP.FCTP.param
                    param.set(param.max_iter, 1)
                    param.set(param.reset, False)  # Set reset
                    param.set(param.num_runs, 1)
                    param.set(param.max_before_diversify, diversify)
                    param.set(param.iter_to_diversify, 5)
                    fctp.solve()
                    mean.append(fctp.solution.tot_cost)
                    print(data_file + ': ' + str(mean))
                res.append(np.mean(mean))
            df[data_file] = res
    df.to_latex(buf=(dir + 'max_before_diversify_hard'))

def test_diversification_iter(ini_files, files, search_range, num_runs):
    df = pd.DataFrame({'Iterations with no improvement before diversification producedure is initiated': search_range})
    for ini_file in ini_files:
        for data_file in files:
            res = []
            for diversify in search_range:
                mean = []
                for run in range(num_runs):
                    fctp = extFCTP.extFCTP(data_file=data_file, ini_file=ini_file)
                    param = extFCTP.FCTP.param
                    param.set(param.max_iter, 300)
                    param.set(param.reset, False)  # Set reset
                    param.set(param.num_runs, 1)
                    param.set(param.max_before_diversify, 20)
                    param.set(param.iter_to_diversify, diversify)
                    fctp.solve()
                    mean.append(fctp.solution.tot_cost)
                    print(data_file + ': ' + str(mean))
                res.append(np.mean(mean))
            df[data_file] = res
    df.to_latex(buf=(dir + 'divers_test_hard'))


if __name__ == "__main__":
    files = ['N207.FCTP', 'N307.FCTP', 'N507.FCTP', 'N1007.FCTP', 'N2007.FCTP','N310E.FCTP', 'N350E.FCTP', 'N3001.FCTP',
             'N3101.FCTP', 'N3201.FCTP', 'N3301.FCTP', 'N3401.FCTP', 'N3501.FCTP', 'N3601.FCTP', 'N3604.FCTP',
             'N3304.FCTP', 'N3507.FCTP', 'N3704.FCTP']
    ini_files = ['ILSHist.ini', 'ILSStandard']
    test_instances(ini_files, files, 1)