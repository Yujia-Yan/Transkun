import matplotlib
import numpy as np
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser(description='plot the empirical cumulative distribution function on onset/offset deviations' )

parser.add_argument('evalJsons', nargs= '+', help = "a seqeunce of the output json files from the computeMetrics script, the deviation output should be enabled")
parser.add_argument('--labels', nargs= '*', help = "specify labels to show on the legend")

parser.add_argument('--offset', action='store_true', help="plot the offset deviation curve. If not specified, onset deviation curve will be plotted")
parser.add_argument('--T', default = 50, type = float, help="time limit(ms), default: 50ms")
parser.add_argument("--output",  nargs="?", help = "filename to save")
parser.add_argument('--noDisplay', action='store_true', help="Do not show the figure.")

args = parser.parse_args()

jsonList = args.evalJsons

plotOffset = args.offset
T = args.T
output = args.output


legends = args.labels
if len(legends)>0:
    if len(legends) != len(jsonList):
        print("Number of labels should match the number of evalJsons.") 
        exit(1)

else:
    legends = jsonList

plt.yticks(np.arange(0,1, 0.05))
plt.xticks(np.arange(0,T, T/10))
plt.xlim(0, T)
plt.grid()
if plotOffset:
    plt.xlabel("Offset Deviation (ms)")
else:
    plt.xlabel("Onset Deviation (ms)")
plt.ylabel("Cumulative Probability")

for jsonFile, legend in zip(jsonList, legends):
    print(jsonFile)
    with open(jsonFile, 'r') as f:
        details = json.load(f)["detailed"]

        devs = sum([_["metrics"]["deviations"] for _ in details], [])
        devs = np.array(devs)
        if plotOffset:
            devs = devs[:, 0]
        else:
            devs = devs[:, 1]
        
        devs = np.abs(devs)
    sns.ecdfplot(1000*devs, legend= legend)

plt.legend(title='', loc='lower right', labels=legends)
if output is not None:
    plt.savefig(output, dpi = 300)
if not args.noDisplay:
    plt.show()

