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
parser.add_argument('--cumulative', action='store_true', help="plot the empirical cumulative density.")
parser.add_argument('--absolute', action='store_true', help="use absolute deviation.")
parser.add_argument('--targetPitch', required=False, type=int,help="only plot specific number.")

args = parser.parse_args()

jsonList = args.evalJsons

plotOffset = args.offset
T = args.T
output = args.output

targetPitch = args.targetPitch

legends = args.labels
if len(legends)>0:
    if len(legends) != len(jsonList):
        print("Number of labels should match the number of evalJsons.") 
        exit(1)

else:
    legends = jsonList

plt.yticks(np.arange(0,1, 0.05))
plt.xticks(np.arange(-T,T, T/10))
plt.xlim(-T, T)
plt.grid()
if plotOffset:
    plt.xlabel("Offset Deviation (ms)")
else:
    plt.xlabel("Onset Deviation (ms)")

if args.cumulative:
    plt.ylabel("Cumulative Probability")
else:
    plt.ylabel("Probability Density")

for jsonFile, legend in zip(jsonList, legends):
    print(jsonFile)
    with open(jsonFile, 'r') as f:
        details = json.load(f)["detailed"]

        devs = sum([_["metrics"]["deviations"] for _ in details], [])
        devs = np.array(devs)

        pitch = devs[:, 0]
        devs = devs[:,1:]

        if plotOffset:
            devs = devs[:, 1]
        else:
            devs = devs[:, 0]
        
        if targetPitch is not None:
            devs = devs[pitch==targetPitch]

        if args.absolute:
            devs = np.abs(devs)

    if args.cumulative:
        sns.ecdfplot(1000*devs, legend= legend, gridsize=8000)
    else:
        sns.kdeplot(1000*devs, legend= legend, gridsize=8000)

plt.legend(title='', loc='upper left', labels=legends)
if output is not None:
    plt.savefig(output, dpi = 300)
if not args.noDisplay:
    plt.show()

