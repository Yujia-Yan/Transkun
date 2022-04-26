import torch
import sys
import os

import argparse

import pathlib
import Evaluation

import collections

from multiprocessing import Pool
import Data
import os
import warnings
import glob

import numpy as np
import json
import itertools


def eval(args):
    path, estPath, gtPath, extendSustainPedal, computeDeviations = args

    audioName = str(path.relative_to(estPath))

    # print(audioName)

    targetPath = gtPath/path.relative_to(estPath)
    # print(path)
    # print(targetPath)
    notesEst = Data.parseMIDIFile(str(path), extendSustainPedal=False)
    notesGT = Data.parseMIDIFile(str(targetPath), extendSustainPedal=extendSustainPedal)


    metrics = Evaluation.compareTranscription(notesEst, notesGT, splitPedal=True, computeDeviations= computeDeviations)

    # for name in metrics:
    # print(metrics, audioName)

    return metrics,audioName
    

if __name__=="__main__":

    argParser = argparse.ArgumentParser(description = 
            "compute metrics directly from MIDI files." + os.linesep+\
            "Note that estDIR should have the same folder structure as the groundTruthDIR." + os.linesep +\
            "The MIDI files to evaluate should have the same extension as the ground truth."+ os.linesep +\
            "Metrics outputed are ordered by precision, recall, f1, overlap." , formatter_class =argparse.RawTextHelpFormatter)

    argParser.add_argument("estDIR")
    argParser.add_argument("groundTruthDIR")
    argParser.add_argument("--outputJSON", help="path to save the output file for detailed metrics per audio file")
    argParser.add_argument("--noPedalExtension", action='store_true', help = "Do not perform pedal extension according to the sustain pedal for the ground truch")
    argParser.add_argument("--nProcess", nargs="?", type=int, default = 1, help = "number of workers for multiprocessing")
    argParser.add_argument("--computeDeviations", action='store_true', help = "output detailed onset/offset deviations for each matched note.")


    warnings.filterwarnings('ignore', module='mir_eval')

    args = argParser.parse_args()


    estPath = args.estDIR
    gtPath = args.groundTruthDIR

    estPath = pathlib.Path(estPath)
    gtPath = pathlib.Path(gtPath)

    outputJSON= args.outputJSON
    extendPedal = not args.noPedalExtension
    computeDeviations = args.computeDeviations
    nProcess = args.nProcess

    filenames = list(estPath.glob(os.path.join('**','*.midi')))


    import tqdm
    if nProcess>1:
        with Pool(nProcess) as p:
            metricsAll = list(
                    tqdm.tqdm(
                    p.imap_unordered(eval, [(_, estPath, gtPath, extendPedal, computeDeviations) for _ in filenames]),
                    total = len(filenames)
                    ))
    else:
            metricsAll = list(
                    tqdm.tqdm(
                    map(eval, [(_, estPath, gtPath, extendPedal, computeDeviations) for _ in filenames]),
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
            continue
        tmp = np.array(aggDict[key])
        resultAgg[key] = (np.mean(tmp , axis = 0).tolist())
    

    for key in resultAgg:
        print( "{}: {}".format(key, resultAgg[key]))

    if outputJSON is not None:
        # torch.save(metricsAll, outputSheetName)
        resultList = [{"name":name, "metrics":m} for m, name in metricsAll]

        result = {"aggregated": resultAgg, "detailed": resultList}



        with open(outputJSON, 'w') as f:
            f.write(json.dumps(result, indent= '\t'))


