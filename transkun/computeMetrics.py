import torch
import sys
import os

import argparse

import pathlib
from . import Evaluation

import collections

from multiprocessing import Pool
from . import Data
import os
import warnings
import glob

import numpy as np
import json
import itertools
import random
import statistics
import scipy


def eval(args):
    path, estPath, gtPath, extendSustainPedal, computeDeviations, pedalOffset, alignOnset, dither, extendPedalEst, onsetTolerance = args

    audioName = str(path.relative_to(estPath))

    # print(audioName)

    targetPath = gtPath/path.relative_to(estPath)
    # print(path)
    # print(targetPath)
    notesEst = Data.parseMIDIFile(str(path), extendSustainPedal=extendPedalEst)
    notesGT = Data.parseMIDIFile(str(targetPath), extendSustainPedal=extendSustainPedal, pedal_ext_offset = pedalOffset)


    metrics = Evaluation.compareTranscription(notesEst, notesGT, splitPedal=True, computeDeviations= computeDeviations, onset_tolerance = onsetTolerance)

    # realign
    onsetDev = [d[1] for d in metrics["deviations"]]
    offsetDev = [d[2] for d in metrics["deviations"]]
    meanOnsetDev = sum(onsetDev)/len(onsetDev)
    meanOffsetDev = sum(offsetDev)/len(offsetDev)


    medianOnsetDev = statistics.median(onsetDev)
    maxDevOnset = max(max(onsetDev), -min(onsetDev))

    if alignOnset:
        
        for n in notesGT:
            n.start += maxDevOnset - medianOnsetDev
            n.end += maxDevOnset -medianOnsetDev

        # r = (random.random()*2-1)*0.005

        for n in notesEst:
            n.start += maxDevOnset 
            n.end += maxDevOnset 

    if dither != 0.0:
        for n in notesGT:
            n.start += dither
            n.end += dither 


        for n in notesEst:
            r = (random.random()*2-1)*dither
            n.start += dither + r
            n.end += dither+ r
        notesEst = Data.resolveOverlapping(notesEst)


        # recompute
    metrics = Evaluation.compareTranscription(notesEst, notesGT, splitPedal=True, computeDeviations= computeDeviations)

    # for name in metrics:
    # print(metrics, audioName)

    return metrics,audioName
    

def main():

    argParser = argparse.ArgumentParser(description = 
            "compute metrics directly from MIDI files." + os.linesep+\
            "Note that estDIR should have the same folder structure as the groundTruthDIR." + os.linesep +\
            "The MIDI files to evaluate should have the same extension as the ground truth."+ os.linesep +\
            "Metrics outputed are ordered by precision, recall, f1, overlap." , formatter_class =argparse.RawTextHelpFormatter)

    argParser.add_argument("estDIR")
    argParser.add_argument("groundTruthDIR")
    argParser.add_argument("--outputJSON", help="path to save the output file for detailed metrics per audio file")
    argParser.add_argument("--noPedalExtension", action='store_true', help = "Do not perform pedal extension according to the sustain pedal for the ground truch")
    argParser.add_argument("--applyPedalExtensionOnEstimated", action='store_true', help = "perform pedal extension for the estimated midi")
    argParser.add_argument("--nProcess", nargs="?", type=int, default = 1, help = "number of workers for multiprocessing")
    argParser.add_argument("--computeDeviations", action='store_true', help = "output detailed onset/offset deviations for each matched note.")
    argParser.add_argument("--alignOnset", action='store_true', help = "whether or not realign the onset.")
    argParser.add_argument("--dither", default = 0.0, type = float, help = "amount of noise added to the prediction.")
    argParser.add_argument("--pedalOffset", default=0.0, type=float, help = "offset added to the groundTruth sustain pedal when extending notes")
    argParser.add_argument("--onsetTolerance", default=0.05, type=float)


    warnings.filterwarnings('ignore', module='mir_eval')

    args = argParser.parse_args()


    estPath = args.estDIR
    gtPath = args.groundTruthDIR

    estPath = pathlib.Path(estPath)
    gtPath = pathlib.Path(gtPath)

    outputJSON= args.outputJSON
    extendPedal = not args.noPedalExtension
    extendPedalEst = args.applyPedalExtensionOnEstimated
    computeDeviations = args.computeDeviations
    nProcess = args.nProcess
    pedalOffset = args.pedalOffset
    alignOnset = args.alignOnset
    dither = args.dither
    onsetTolerance = args.onsetTolerance

    filenames = list(estPath.glob(os.path.join('**','*.midi')))+ list(estPath.glob(os.path.join('**','*.mid')))

    filenamesFiltered = []

    for filename in filenames:
        targetPath = gtPath/filename.relative_to(estPath)
        if targetPath.exists():
            filenamesFiltered.append(filename)

    filenames = filenamesFiltered


    import tqdm
    if nProcess>1:
        with Pool(nProcess) as p:
            metricsAll = list(
                    tqdm.tqdm(
                    p.imap_unordered(eval, [(_, estPath, gtPath, extendPedal, computeDeviations, pedalOffset, alignOnset, dither, extendPedalEst, onsetTolerance) for _ in filenames]),
                    total = len(filenames)
                    ))
    else:
            metricsAll = list(
                    tqdm.tqdm(
                    map(eval, [(_, estPath, gtPath, extendPedal, computeDeviations, pedalOffset, alignOnset, dither, extendPedalEs, onsetTolerancet) for _ in filenames]),
                    total = len(filenames)
                    ))




    # aggregate
    aggDict =collections.defaultdict(list)

    for m, _ in metricsAll:
        for key in m:
            aggDict[key].append(m[key])


    resultAgg = dict()
    for key in aggDict:
        # aggDict
        # tmp = torch.Tensor(aggDict[key])
        # result[key] = tmp.mean(dim = 0).list()
        if key == "deviations":
            # perform normality test
            devAll = sum(aggDict[key], [])
            dev_onset = np.array([_[1] for _ in devAll])
            dev_offset = np.array([_[2] for _ in devAll])
            onset_result =  scipy.stats.anderson(dev_onset)
            offset_result=  scipy.stats.anderson(dev_offset)
            resultAgg["deviation_onset_normality"]= onset_result.statistic
            resultAgg["deviation_offset_normality"]= offset_result.statistic

        else:
            tmp = np.array(aggDict[key])
            resultAgg[key] = (np.mean(tmp , axis = 0).tolist())
    
    # perform normality test for deviations

    for key in resultAgg:
        print( "{}: {}".format(key, resultAgg[key]))

    if outputJSON is not None:
        # torch.save(metricsAll, outputSheetName)
        resultList = [{"name":name, "metrics":m} for m, name in metricsAll]

        result = {"aggregated": resultAgg, "detailed": resultList}



        with open(outputJSON, 'w') as f:
            f.write(json.dumps(result, indent= '\t'))


if __name__=="__main__":
    main()
