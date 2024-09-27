import mir_eval
import numpy as np


from . import Data

def _listOfListToTuple(nested_list):
    return [tuple(l) for l in nested_list]

def compareBracket(intervalEst, intervalGT):
    nGT = len(intervalGT)
    nEst = len(intervalEst)

      
    nUnion = len(set(_listOfListToTuple(intervalEst+ intervalGT)))
    nCorrect = nGT+nEst-nUnion

    return nGT, nEst, nCorrect
    

def intersectTwoInterval(intervalA, intervalB):
    l = max(intervalA[0], intervalB[0])
    r = min(intervalA[1], intervalB[1])
    return (l,r)

def findIntersectListOfIntervals(listA, listB):
    i = 0
    j = 0
    result = []
    while i<len(listA) and j<len(listB):
        l,r = intersectTwoInterval(listA[i], listB[j])
        if r>=l:
            # check if (l,r) can be merged into the previous one
            if len(result)>0 and result[-1][1] == l:
                result[-1] = (result[-1][0],r)
            else:
                result.append((l,r))
        
        if listA[i][1] < listB[j][1]:
            i = i+1
        else:
            j = j+1

        
    
    return result


    
def computeIntervalLengthSum(intervals, countZero=True):
    s = 0
    if countZero:
        prevEnd = -1
        for e in intervals:
            s+= e[1]-e[0]
            if prevEnd < e[0]:
                s+= 1

            prevEnd = e[1]
    else:
        for e in intervals:
            s+= e[1]-e[0]

    return s


def compareFramewise(intervalEst, intervalGT, countZero=True):
    nEst = computeIntervalLengthSum(intervalEst, countZero)
    nGT = computeIntervalLengthSum(intervalGT, countZero)
    intersected = findIntersectListOfIntervals(intervalEst,intervalGT)
    nIntersected = computeIntervalLengthSum(intersected, countZero)
    nUnion = nGT+nEst- nIntersected

    return nGT,nEst, nIntersected




def midi_to_freq(midi):
    if midi>=0:
        freq = 2**((midi -69)/12)*440
    else:
        # tricks for pedals
        freq = 2**((-midi -69)/12)*440*100
    return freq
def getSpan(eventList):
    r = max([e.end for e in eventList])
    return r


def computeFrameScore(estimated, gt, eventTypes):
    # by default step is 0.01 to make it aligned with most papers

    # len(eventTypes)

    # eventTypes = set([e.pitch for e in estimated] +
    # [gt.pitch for e in estimated])

    # originally skiiping empty groundtruth
    # if len(estimated)==0 or len(gt)==0:
        # return 0,0,0




    intervalsA = Data.prepareIntervalsNoQuantize(estimated, eventTypes)["intervals"]
    intervalsB = Data.prepareIntervalsNoQuantize(gt,  eventTypes)["intervals"]
    assert(len(intervalsA) == len(eventTypes))
    assert(len(intervalsB) == len(eventTypes))

    nGT = 0
    nEst = 0
    nCorrect = 0

    for IA, IB in zip(intervalsA, intervalsB):

        cur_nGT, cur_nEst, cur_nCorrect = compareFramewise(IA, IB, countZero=False)
        # print(compareFramewise(IA, IB))
        nGT += cur_nGT
        nEst += cur_nEst
        nCorrect += cur_nCorrect

    p = nCorrect/(nEst+1e-8)
    r = nCorrect/(nGT+1e-8)
    f = 2*nCorrect/(nEst+ nGT+1e-8)
    o = nCorrect/(nEst+nGT-nCorrect+1e-8)

    return p,r,f,o
        


def compareMatchedDeviations(estimated, gt, splitPedal):
    resultEst, pedalEst = prepareDataForEvaluation(estimated, splitPedal=splitPedal)
    resultGT, pedalGT= prepareDataForEvaluation(gt, splitPedal=splitPedal)

    metrics = dict()


    matched  =  mir_eval.transcription.match_notes(
            resultGT["intervals"],
            resultGT["pitches"],
            resultEst["intervals"],
            resultEst["pitches"],
            onset_tolerance = 0.1,
            offset_min_tolerance = 0.1
            )

    
    # compute deviations 
    deviations = []
    for idxGT, idxEst in matched:
        intervalGT = resultGT["intervals"][idxGT]
        intervalEST = resultEst["intervals"][idxEst]
        curDiff = intervalGT-intervalEST
        deviations.append(curDiff)

    return deviations 


def compareTranscription(estimated, gt, splitPedal=False, computeDeviations = False, **kwargs):


    # convert data into the format mir_eval use 

    resultEst, pedalEst = prepareDataForEvaluation(estimated, splitPedal=splitPedal)
    resultGT, pedalGT= prepareDataForEvaluation(gt, splitPedal=splitPedal)

    # print(resultEst)

    # compute framewise (pitch activation level) note level score (onest, onset+ offset, onset+offset+velocity) also evaluate pedal individually

    metrics = dict()
    # 1. framewise score (pitch activation level score)
    # computeFrameScore


    metrics["frame"] = computeFrameScore(estimated, gt, eventTypes = list(range(21,108+1)))
     

    
    nGT = resultGT["intervals"].shape[0]
    nEst = resultEst["intervals"].shape[0]


    # 2. note onset 


    metrics["note"] =  mir_eval.transcription.precision_recall_f1_overlap(
            resultGT["intervals"],
            resultGT["pitches"],
            resultEst["intervals"],
            resultEst["pitches"],
            offset_ratio = None,
            **kwargs
            )

    metrics["note+velocity"] = mir_eval.transcription_velocity.precision_recall_f1_overlap(
            resultGT["intervals"],
            resultGT["pitches"],
            resultGT["velocities"],
            resultEst["intervals"],
            resultEst["pitches"],
            resultEst["velocities"],
            offset_ratio = None,
            **kwargs
            )
    
    # 3. note onset + offset 
    metrics["note+offset"] =  mir_eval.transcription.precision_recall_f1_overlap(
            resultGT["intervals"],
            resultGT["pitches"],
            resultEst["intervals"],
            resultEst["pitches"],
            **kwargs
            )

    # note onset + offset + velocity
    metrics["note+velocity+offset"] = mir_eval.transcription_velocity.precision_recall_f1_overlap(
            resultGT["intervals"],
            resultGT["pitches"],
            resultGT["velocities"],
            resultEst["intervals"],
            resultEst["pitches"],
            resultEst["velocities"],
            **kwargs
            )
    
    metrics["nGT"] = nGT
    metrics["nEst"] = nEst

    # deviations of matched notes
    if computeDeviations:
        matched  =  mir_eval.transcription.match_notes(
                resultGT["intervals"],
                resultGT["pitches"],
                resultEst["intervals"],
                resultEst["pitches"],
                onset_tolerance = 0.8,
                offset_min_tolerance = 0.8
                )

        # compute deviations 
        deviations = []
        for idxGT, idxEst in matched:
            intervalGT = resultGT["intervals"][idxGT]
            intervalEST = resultEst["intervals"][idxEst]
            pitches_midi  = int(resultEst["pitches_midi"][idxEst])

            curDiff = intervalGT-intervalEST
            deviations.append([pitches_midi]+curDiff.tolist())

        metrics["deviations"] = deviations


    if len(pedalEst)>0:
        # evaluate pedals

        for cc in pedalEst:
            curEst = pedalEst[cc]
            curGT = pedalGT[cc]
            nGTPedal = curGT["intervals"].shape[0]
            nEstPedal = curEst["intervals"].shape[0]

            if nGTPedal>0:
                metrics["pedal"+str(cc)+"frame"] = computeFrameScore(estimated, gt, eventTypes =[-cc] )
                metrics["pedal"+str(cc)] =  mir_eval.transcription.precision_recall_f1_overlap(
                        curGT["intervals"],
                        curGT["pitches"],
                        curEst["intervals"],
                        curEst["pitches"],
                        offset_ratio = None,
                        **kwargs
                        )

                metrics["pedal"+str(cc) +"+offset"] =  mir_eval.transcription.precision_recall_f1_overlap(
                        curGT["intervals"],
                        curGT["pitches"],
                        curEst["intervals"],
                        curEst["pitches"],
                        **kwargs
                        )

                metrics["pedal"+str(cc)+"nGT"] = nGTPedal       
                metrics["pedal"+str(cc)+"nEst"] = nEstPedal       
                


    # each entry has (precision, recall, f1, average overlap ratio)

    return metrics





def prepareDataForEvaluation(notes, ccList = [64,67], splitPedal=False):
    # convert notes to 
    # intervals: np.ndarray shape = (n, 2) 
    # pitches: np.ndarray shape = (n,),  in Hz
    # velocities: np.ndarray shape=(n,) between 0- 127

    

    # filter out unsupported symbols
    notes = [n for n in notes if -n.pitch in ccList or n.pitch>=0]


    if splitPedal:
        intervals = np.array([[n.start, n.end] for n in notes if n.pitch>=0])
        pitches = np.array([midi_to_freq(n.pitch) for n in notes if n.pitch>=0])
        pitches_midi = np.array([n.pitch for n in notes if n.pitch>=0])
        velocities = np.array([n.velocity for n in notes if n.pitch>=0])
    else:
        intervals = np.array([[n.start, n.end] for n in notes])
        pitches = np.array([midi_to_freq(n.pitch) for n in notes])
        pitches_midi = np.array([n.pitch for n in notes])
        velocities = np.array([n.velocity for n in notes])

    if intervals.shape == (0,):
        intervals = np.zeros(shape= (0,2))
    # for pedal, we group all pedals individually

    pedals =  dict()

    for cc in ccList:

        intervals_pedal = np.array([[n.start, n.end] for n in notes if n.pitch==-cc])
        pitches_pedal = np.array([1 for n in notes if n.pitch==-cc])
        velocities_pedal = np.array([n.velocity for n in notes if n.pitch==-cc])

        if intervals_pedal.shape == (0,):
            intervals_pedal = np.zeros(shape= (0,2))

        curResult = { "intervals": intervals_pedal,
          "pitches": pitches_pedal,
          "velocities": velocities_pedal
          }
        pedals[cc] = curResult
    
    result = { "intervals": intervals,
      "pitches": pitches,
      "pitches_midi": pitches_midi,
      "velocities": velocities
      }

    return result, pedals
     

