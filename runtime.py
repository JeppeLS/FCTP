import extFCTP
import time


fctp = extFCTP.extFCTP(data_file="N3704.FCTP", ini_file='ILSHist.ini')
param = extFCTP.FCTP.param
param.set(param.max_iter, 50)
param.set(param.num_runs, 1)
param.set(param.max_before_diversify, 10)
param.set(param.iter_to_diversify, 5)
param.set(param.max_no_imp, 500)

start_our = time.time()
fctp.solve()
time_our = time.time() - start_our

fctp = extFCTP.extFCTP(data_file="N3704.FCTP", ini_file='ILSStandard.ini.ini')
param = extFCTP.FCTP.param
param.set(param.max_iter, 50)
param.set(param.num_runs, 1)
param.set(param.max_no_imp, 500)

start_klose = time.time()
fctp.solve()
time_klose = time.time() - start_klose

print('Our: ' + str(time_our))
print('Klose: ' + str(time_klose))