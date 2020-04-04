"""
    Class "FCTPy.param" belonging to package FCTPy.
    
    Defines parameters and methods to adjust them.
    
    author:
        Andreas Klose
    version:
        1.2 (02/03/20)
"""

from configparser import ConfigParser

# Identifiers for the diverse parameters
improve_method = 1 
greedy_meas = 2
ls_type = 3
max_iter = 4
max_no_imp = 5
gls_alpha_fcost = 6
gls_alpha_tcost = 7
sa_cool_beta = 8
min_acc_rate = 9
ini_acc_rate = 10
sample_growth = 11
num_runs = 12
do_restart = 13
what_out = 14
screen = 15
pop_size = 16
num_childs = 17
ils_rep = 18
rtr_percent = 19
cpx_time = 20
cpx_nodelim = 21
benders = 22
callbck = 23
cpx_localbranch = 24
cpx_cut_aggress = 25
cpx_dis_cut = 26
cpx_use_SNF_Gomory = 27
ils_type = 28
kstep = 29
max_before_reset = 30
tolerance = 100
init_file = 200;
input_file = 201
output_file = 202

# Constants describing possible parameter values  
first_accept = 0
best_accept = 1
greedy = 0
lpheu = 1
no_greedy = 0
greedy_lin_cap = 1
greedy_lin_rem_cap = 2
greedy_lin_totcost = 3
no_detail = 1
detailed = 2
off = 0
on = 1
no = 0
yes = 1
ls = 1
ils = 2
msls = 3
sa = 4
sa_osman = 5
gls = 6
vns = 7
rtr = 8
rtr_jeanne=9
grasp = 10
ants = 11
ea = 12
ts = 13
rtr_ils = 14
ils_rtr = 15
rtr_vns = 16
vns_rtr = 17
extended_ss = 18
alter_ss = 19
cpx_opt = 20
tol_default = 1.0E-4

ils_standard = 0 # Vanilla ILS as implemented by Andreas Klose
ils_random = 1 # ILS with weighted randomised accent with no reset
ils_random_reset = 2 # ILS with weighted randomised accent with reset



# Parameters, initially set to default values
__param = { improve_method:None,\
            greedy_meas:no_greedy,\
            ls_type:first_accept,\
            ils_type:ils_standard,\
            max_iter:50,\
            kstep: 3,
            max_before_reset: 100,
            max_no_imp:100,\
            gls_alpha_fcost:0.1,\
            gls_alpha_tcost:0.0,\
            sa_cool_beta:0.95,\
            min_acc_rate:0.001,\
            ini_acc_rate:0.3,\
            sample_growth:0.02,\
            num_runs:1,\
            do_restart:yes,\
            what_out:no_detail,\
            screen:on,\
            pop_size:100,\
            num_childs:100,\
            ils_rep:20,\
            rtr_percent:0.1,\
            cpx_time:1.0E30,\
            cpx_nodelim:2147483647,\
            benders:no,\
            callbck:no,\
            cpx_localbranch: no,\
            cpx_cut_aggress: no,\
            cpx_dis_cut: no,\
            cpx_use_SNF_Gomory: no,\
            tolerance:tol_default,\
            init_file:None,\
            input_file:None,\
            output_file:None}             

# Parameter default values
__default = { improve_method:None,\
            greedy_meas:no_greedy,\
            ls_type:first_accept,\
            ils_type:ils_standard,\
            max_iter:50,\
            kstep: 3,
            max_before_reset: 100,
            max_no_imp:100,\
            gls_alpha_fcost:0.1,\
            gls_alpha_tcost:0.0,\
            sa_cool_beta:0.95,\
            min_acc_rate:0.001,\
            ini_acc_rate:0.3,\
            sample_growth:0.02,\
            num_runs:1,\
            do_restart:yes,\
            what_out:no_detail,\
            screen:on,\
            pop_size:100,\
            num_childs:100,\
            ils_rep:20,\
            rtr_percent:0.1,\
            cpx_time:1.0E30,\
            cpx_nodelim:2147483647,\
            benders:no,\
            callbck:no,\
            cpx_localbranch: no,\
            cpx_cut_aggress: no,\
            cpx_dis_cut: no,\
            cpx_use_SNF_Gomory: no,\
            tolerance:tol_default,\
            init_file:None,\
            input_file:None,\
            output_file:None}

# Parameter names used in configuration file
__para_name = { improve_method:"ImproveMethod",\
                greedy_meas:"GreedyMeasure",\
                ls_type:"LocalSearch",\
                ils_type:"ILSType",\
                max_iter:"MaxIter",\
                max_no_imp:"MaxIterWithoutImprove", \
                kstep: "kstep",
                max_before_reset: "MaxIterBeforeReset",
                gls_alpha_fcost:"GLS_alpha_fixedcost",\
                gls_alpha_tcost:"GLS_alpha_transpcost",\
                sa_cool_beta:"SA_beta",\
                min_acc_rate:"min_acc_rate",\
                ini_acc_rate:"ini_acc_rate",\
                sample_growth:"SA_sample_growth",\
                num_runs:"Runs",\
                do_restart:"Restart",\
                what_out:"Output",\
                screen:"Intermediate_Output",\
                pop_size:"lambda",\
                num_childs:"mu",\
                ils_rep:"RTR_ILS_REP",\
                rtr_percent:"RTR_percent", \
                cpx_time:"CPXTIME",\
                cpx_nodelim:"CPXNODELIM",\
                benders:"BENDERS",\
                callbck:"CALLBCK",\
                cpx_localbranch:"LOCBRANCH",\
                cpx_cut_aggress:"CUTS_AGGRESSIVE",\
                cpx_dis_cut: "DISJUNCTIVE_CUTS",\
                cpx_use_SNF_Gomory: "SNF_GOMORY",\
                tolerance:"Tolerance",\
                init_file:"Ini file",\
                input_file:"Input file",\
                output_file:"Output file"}

def set_defaults( ):
    """
    Sets all parameters to default values.
    """
    __param=__default

def set( i, value ):
    """
    Sets parameter with identifier "i" to the value "value"
    """
    if i in __param: __param[i] = value

def get( i ):
    """
    Return value of parameter with identifier "i".
    """
    return __param.get(i,None)

def get_proc_name ( i ):
    """
    Return name of the improvement procedure that has identifier no. i
    """
    if i==ls: return "Local Search"
    if i==ils: return "Iterated Local Search"
    if i==ils_rtr: return "Hybrid ILS-RTR"
    if i==msls: return "Multi-Start Local Search"
    if i==sa: return "Simulated Annealing"
    if i==sa_osman: return "SA a la Osman"
    if i==gls: return "Guided Local Search"
    if i==vns: return "Variable Neighbourhood Search"
    if i==vns_rtr: return "Hybrid VNS-RTR"
    if i==rtr: return "Record-to-Record Travel"
    if i==rtr_jeanne: return "Jeanne's Record-to-Record Travel"
    if i==rtr_ils: return "Hybrid RTR_ILS"
    if i==rtr_vns: return "Hybrid RTR_VNS"
    if i==grasp: return "GRASP"
    if i==ants: return "Ant Colony"
    if i==ea: return "Evolutionary Algorithm"
    if i==ts: return "Tabu Search"
    if i==extended_ss: return "Extended Scatter Search"
    if i==alter_ss: return "Alternative Scatter Search"
    if i==cpx_opt: 
        if (__param[benders]==0): return "Optimal solution with CPLEX"
        if (__param[benders]==1): return "Optimal solution with CPLEX (automated Benders)"
    return "UNKOWN"

def print_params( ):
    """
    Print the current parameter setting
    """
    print("----------------------------------------------------------")        
    print("FCTP parameter setting:")
    print("----------------------------------------------------------")
    print("Input data file            :",__param[input_file] )
    print("Output is send to file     :",__param[output_file] )
    print("Method to be applied       :",get_proc_name(__param[improve_method] ) )
    if __param[greedy_meas] > 0:
        print("Start solutions obtained by: Greedy" )
        print("Greedy measure used        :",__param[greedy_meas] )
    else:
        print("Start solutions obtained by: LP heuristic" )
    print("Type of local search to use:",__param[ls_type] )
    print("Type of ILS to use:",__param[ils_type] )
    print("Number of iterations       :",__param[max_iter] )
    print("Iterations without improve :",__param[max_no_imp] )
    print("GLS - penalty fixed cost   :",__param[gls_alpha_fcost] )
    print("GLS - penalty transp. cost :",__param[gls_alpha_tcost] )
    print("SA - parameter beta        :",__param[sa_cool_beta] )
    print("SA - initial accept. rate  :",__param[ini_acc_rate] )
    print("SA - final accept. rate    :",__param[min_acc_rate] )
    print("SA - sample size growth    :",__param[sample_growth] )
    print("Output detail              :",__param[what_out] )
    print("Intermediate Output is on  :",__param[screen] )
    print("Number of runs             :",__param[num_runs])
    print("Each run with restart?     :",__param[do_restart])
    print("Population size in EA      :",__param[pop_size])
    print("Number of childs in EA     :",__param[num_childs])
    print("ILS iterations in RTR-ILS  :",__param[ils_rep])
    print("RTR-threshold precentage   :",__param[rtr_percent])
    print("CPLEX time limit           :",__param[cpx_time] )
    print("CPLEX node limit           :",__param[cpx_nodelim] )
    print("CPLEX - local branching    :",__param[cpx_localbranch])
    print("CPLEX - aggressive cuts    :",__param[cpx_cut_aggress])
    print("CPLEX - disjunctive cuts   :",__param[cpx_dis_cut])
    print("CPLEX - SNF Gomory         :",__param[cpx_use_SNF_Gomory])
    print("Benders decomposition      :",__param[benders])
    print("Callback heuristic         :",__param[callbck] )
    print("----------------------------------------------------------") 

def __is_number(s):
    "Returns true if the string s is a number"
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_ini_file( ):
    """
    Read the initialization file given as __param['ini_file']
    """
    pkeys = {}
    for p in __para_name: pkeys[__para_name[p].lower()] = p
    set_defaults()
    iniFile = __param[init_file]
    cfg = ConfigParser()
    if iniFile and cfg.read(iniFile):
        for sect in cfg.sections():
            plist = [pname for pname in cfg[sect] if pname in pkeys]
            for pname in plist:
                v = cfg['PARAMETERS'][pname]
                p = pkeys[pname]
                __param[p]= ( float(v) if '.' in v else int(v)) if __is_number(v) else v 
    print_params()
