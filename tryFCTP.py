from extFCTP import extFCTP
from sys import argv
from getopt import getopt
from tempfile import NamedTemporaryFile
from FCTP.convert import convert
import numpy as np


def __printHelp():
    print("Usage: ", argv[0] + " data_file <-i ini_file> <-f format>")
    print("    Options:")
    print("        -i ini_file: Use ini_file instead of FCTPheur.ini as ini-file")
    print("        -f format  : Use a different input data format.")
    print("                     Format can be one of the following:")
    print("                     A: use Agarwal/Aneja input data format")
    print("                     G: use original Glover input data format")
    print("                     R: use Roberti's input data format")
    exit()


if __name__ == "__main__":

    if len(argv) < 2: __printHelp()
    data_file = argv[1]
    ini_file = None
    in_form = None

    if len(argv) > 2:
        opts, args = getopt(argv[2:], "hi:f:", ["inifile=", "format="])
        for o, a in opts:
            if o in ("-h", "--help"): __printHelp()
            if o in ("-i", "--inifile"): ini_file = a
            if o in ("-f", "--format"): in_form = a

    # If input data format is different from the default, then
    # translate data to a temporary file of standard format
    tmpfil = None
    if not in_form is None:
        tmpfil = NamedTemporaryFile()
        err = convert(data_file, in_form, out_file=tmpfil.name)
        if err > 0:
            print("Cannot convert data format to default one!")
            exit()
        data_file = tmpfil.name

    fctp = extFCTP(data_file=data_file, ini_file=ini_file)
    if fctp.err > 0:
        print("Error:", fctp.err)
        exit()

    # Solve the problem instance using the method selected in the ini-file
    if fctp.solve() == 0:
        fctp.solution.print_flows(lobnd=fctp.lobnd)
        # Draw trajectory of generated objective values for all runs
        if not fctp.all_hist is None:
            import matplotlib.pyplot as plt

            max_iter = max([len(h) for h in fctp.all_hist])
            hist_dat = np.zeros(max_iter, dtype=float)
            for h in fctp.all_hist: hist_dat[:len(h)] += np.array(h)
            nruns = len(fctp.all_hist)
            count = [len([h for h in fctp.all_hist if len(h) > i]) for i in range(max_iter)]
            hist_dat /= np.array(count)
            plt.title("Generated objective function values (average over all runs)")
            plt.plot(hist_dat, marker='o', color='blue', markersize=2)
            plt.show()


