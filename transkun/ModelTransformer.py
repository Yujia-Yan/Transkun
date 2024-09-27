import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint
import math
from .Data import *
from .Util import *
import torch.distributed as dist
from .Evaluation import *
from .LayersTransformer import *
from collections import defaultdict


from . import CRF


class ModelConfig:
    def __init__(self):

        # self.f_min = 20
        # self.f_max = 16000
        # self.n_mels = 463
        self.f_min = 30
        self.f_max = 8000
        self.n_mels = 229

        self.segmentHopSizeInSecond = 8
        self.segmentSizeInSecond = 16

        self.hopSize = 1024
        self.windowSize = 4096
        self.fs = 44100
        self.nExtraWins = 5

        # self.baseSize = 64
        self.baseSize = 40
        self.downsampleF = True

        self.posEmbedInitGamma = 1

        self.nHead = 4
        self.fourierSize = 64

        self.nLayers = 6
        # self.enabledAttn =["F", "T", "All0", "0All"] 
        self.enabledAttn =["F", "T"]
        # self.enabledAttnCross =["F",  "All0", "0All"] 
        self.hiddenFactorAttn= 1
        self.hiddenFactor = 4

        self.velocityPredictorHiddenSize = 512
        self.refinedOFPredictorHiddenSize = 512

        self.scoringExpansionFactor = 4
        self.useInnerProductScorer = True


        self.scoreDropoutProb = 0.1
        self.contextDropoutProb = 0.1
        self.velocityDropoutProb = 0.1
        self.refinedOFDropoutProb= 0.1

    def __repr__(self):
        return repr(self.__dict__)

Config = ModelConfig


class TransKun(torch.nn.Module):
    Config = ModelConfig
    def __init__(self, conf):
        super().__init__()

        self.conf = conf
        self.hopSize = conf.hopSize

        self.windowSize = conf.windowSize
        self.fs = conf.fs

        self.segmentSizeInSecond = conf.segmentSizeInSecond
        self.segmentHopSizeInSecond = conf.segmentHopSizeInSecond




        self.framewiseFeatureExtractor = MelSpectrum(conf.windowSize,
                                                     f_min = conf.f_min,
                                                     f_max = conf.f_max,
                                                     n_mels = conf.n_mels,
                                                     fs = conf.fs,
                                                     nExtraWins = conf.nExtraWins,
                                                     log=True,
                                                     toMono=True
                                                     )
    

        self.targetMIDIPitch= [-64, -67] + list(range(21, 108+1))

        useInnerProductScorer = conf.useInnerProductScorer
        self.useInnerProductScorer = useInnerProductScorer
        if useInnerProductScorer:
            self.scorer = ScaledInnerProductIntervalScorer(
                    conf.baseSize*conf.scoringExpansionFactor,
                    1,
                    dropoutProb = conf.scoreDropoutProb)
        else:
            from .Layers_ablation import PairwiseFeatureBatch
            self.scorerProj = nn.Linear(len(self.targetMIDIPitch)*conf.baseSize, 512)
            self.scorer = PairwiseFeatureBatch(512, outputSize= len(self.targetMIDIPitch) )


        self.velocityPredictor = nn.Sequential(
                                nn.Linear(conf.baseSize*3*conf.scoringExpansionFactor,
                                    conf.velocityPredictorHiddenSize),
                                nn.GELU(),
                                nn.Dropout(conf.velocityDropoutProb),
                                nn.Linear(conf.velocityPredictorHiddenSize, 128)
                                )


        # output dequantize offset + presence logits
        self.refinedOFPredictor = nn.Sequential(
                                nn.Linear(conf.baseSize*3*conf.scoringExpansionFactor,
                                    conf.refinedOFPredictorHiddenSize),
                                nn.GELU(),
                                nn.Dropout(conf.refinedOFDropoutProb),
                                nn.Linear(conf.refinedOFPredictorHiddenSize, 4)
                                )

        
        self.backbone = Backbone(
                            inputSize = self.framewiseFeatureExtractor.nChannel,
                            baseSize = conf.baseSize,
                            posEmbedInitGamma = conf.posEmbedInitGamma,
                            nHead = conf.nHead,
                            fourierSize = conf.fourierSize,
                            hiddenFactor = conf.hiddenFactor,
                            hiddenFactorAttn = conf.hiddenFactorAttn,
                            expansionFactor = conf.scoringExpansionFactor,
                            nLayers = conf.nLayers,
                            dropoutProb = conf.contextDropoutProb,
                            enabledAttn =conf.enabledAttn,
                            downsampleF = conf.downsampleF)

        
    def getDevice(self):
        return next(self.parameters()).device

                

    def processFramesBatch(self, framesBatch):
        
        nBatch = framesBatch.shape[0]
        # nChannel = framesBatch.shape[1]
        # print(framesBatch.shape)
        
        # gain normalization
        # if self.training:
        framesBatchMean = torch.mean(framesBatch, dim = [1,2,3], keepdim=True)
        framesBatchStd = torch.std(framesBatch, dim = [1,2,3], keepdim=True)
        framesBatch = (framesBatch - framesBatchMean)/(framesBatchStd+ 1e-8)


        featuresBatch = self.framewiseFeatureExtractor(framesBatch).contiguous()

        # print(featuresBatch.shape)

        # now with shape [nBatch, nAudioChannel, nStep, NFreq, nChannel]
        nChannel = 1


        featuresBatch  = featuresBatch.view(nBatch*nChannel, *featuresBatch.shape[-3:])
        # now with shape [nBatch* nAudioChannel, nStep, NFreq, nChannel]


        ctx = self.backbone(featuresBatch, outputIndices = torch.tensor(self.targetMIDIPitch, device = featuresBatch.device))

        # construct pairwise score matrix

        # S = torch.einsum("iptd, iptd->"
        # print(ctx.shape)

        # obtain intervalic features
        # print(ctx.shape)

        # # onset:
        # tmp1 = ctx.unsqueeze(-2)

        # # offset
        # tmp2 = ctx.unsqueeze(-3)


        # tmp1, tmp2 = torch.broadcast_tensors(tmp1, tmp2)
        # print(tmp1.shape,tmp2.shape)




        if self.useInnerProductScorer:
            S_batch, S_skip_batch = self.scorer(ctx)

            # now with shape [ nStep, nStep, nBatch, nSym ] 




        else:
            # ctx = ctx.
            ctxScore = ctx.permute(2, 0, 1,3).flatten(-2,-1)
            S_batch, S_skip_batch = self.scorer(self.scorerProj(ctxScore), 10240)



        # batch the CRF together 
        S_batch = S_batch.flatten(-2,-1)
        S_skip_batch= S_skip_batch.flatten(-2,-1)



        # with shape [*, nBatch*nSym]

        crf = CRF.NeuralSemiCRFInterval(S_batch, S_skip_batch)


        return crf, ctx


    def log_prob(self, xBatch, notesBatch):
        nBatch = xBatch.shape[0]
        xBatch = xBatch.transpose(-1,-2)
        # now with shape[nbatch, nAudioChannel, nSample]

        device = xBatch.device


        #[nBatch, nChannel, nSample]
        framesBatch = makeFrame(xBatch, self.hopSize, self.windowSize)

        # crf returned would be nBatch*nChannel flattened for batch processing
        crfBatch, ctxBatch = self.processFramesBatch(framesBatch)


        # prepare groundtruth 
        intervalsBatch= []
        velocityBatch = []
        ofRefinedGTBatch = []
        ofPresenceGTBatch = []
        for notes in notesBatch:
            data = prepareIntervals(notes, self.hopSize/self.fs, self.targetMIDIPitch)
            intervalsBatch .append(data["intervals"])
            velocityBatch.append(sum(data["velocity"], []))
            ofRefinedGTBatch.append(sum(data["endPointRefine"], []))
            ofPresenceGTBatch.append(sum(data["endPointPresence"], []))


        intervalsBatch_flatten = sum(intervalsBatch , [])
        assert( len(intervalsBatch_flatten) == nBatch* len(self.targetMIDIPitch))

        # tmp = torch.Tensor(sum(intervalsBatch_flatten, []))
        # print(tmp.max())


        pathScore = crfBatch.evalPath(intervalsBatch_flatten) 
        logZ = crfBatch.computeLogZ()
        logProb = pathScore - logZ
        logProb = logProb.view(nBatch, -1)


        # then fetch the attrbute features for all intervals

        nIntervalsAll =  sum([len(_) for _ in intervalsBatch_flatten])

        if nIntervalsAll>0:

            ctx_a_all, ctx_b_all, symIdx_all, scatterIdx_all = self.fetchIntervalFeaturesBatch(ctxBatch, intervalsBatch)


            attributeInput = torch.cat([ctx_a_all,
                                       ctx_b_all,
                                       ctx_a_all*ctx_b_all
                                       ],dim = -1)

            
            # prepare groundtruth for velocity

            velocityBatch = sum(velocityBatch, [] )
            ofRefinedGTBatch = sum(ofRefinedGTBatch, [] )
            ofPresenceGTBatch = sum(ofPresenceGTBatch, [])

            logitsVelocity = self.velocityPredictor(attributeInput)
            logitsVelocity = F.log_softmax(logitsVelocity, dim = -1)

            velocityBatch= torch.tensor(velocityBatch, dtype =torch.long, device = device)

            logProbVelocity = torch.gather(logitsVelocity, dim = -1, index = velocityBatch.unsqueeze(-1)).squeeze(-1)

            ofRefinedGTBatch = torch.tensor(ofRefinedGTBatch, device = device, dtype = torch.float)
            ofPresenceGTBatch = torch.tensor(ofPresenceGTBatch, device = device, dtype = torch.float)
            

            # shift it to [0,1]
            # print("GT:", ofRefinedGTBatch)

            ofRefinedGTBatch = ofRefinedGTBatch*0.99+0.5

            ofValue, ofPresence = self.refinedOFPredictor(attributeInput).chunk(2, dim = -1)


            # ofValue = F.logsigmoid(ofValue)

            ofDist = torch.distributions.ContinuousBernoulli(logits=ofValue)

            logProbOF = ofDist.log_prob(ofRefinedGTBatch).sum(-1)

            ofPresenceDist = torch.distributions.Bernoulli(logits = ofPresence)

            logProbOFPresence = ofPresenceDist.log_prob(ofPresenceGTBatch).sum(-1)






            # scatter them back
            logProb = logProb.view(-1)

            
            logProb = logProb.scatter_add(-1, scatterIdx_all, logProbVelocity+ logProbOF + logProbOFPresence)

        logProb = logProb.view(nBatch, -1)

        return logProb

    def computeStatsMIREVAL(self, xBatch, notesBatch):
        nBatch = xBatch.shape[0]
        xBatch = xBatch.transpose(-1,-2)

        device = xBatch.device


        #[nBatch, nChannel, nSample]
        framesBatch = makeFrame(xBatch, self.hopSize, self.windowSize)

        # get the transcription from the frames
        notesEstBatch, _ = self.transcribeFrames(framesBatch)


        assert(len(notesBatch) == len(notesEstBatch))

        # metricsBatch = [Evaluation.compareTranscription(est, gt) for est, gt in zip(notesEstBatch, notesBatch)  ]


        # aggregate by  batch count
        metricsBatch = []
        
        nEstTotal = 0
        nGTTotal = 0 
        nCorrectTotal = 0
          
        for est, gt in zip(notesEstBatch, notesBatch):
            metrics = compareTranscription(est, gt) 
            p,r,f, _ = metrics["note+offset"]

            nGT= metrics["nGT"]
            nEst= metrics["nEst"]

            nCorrect = r*nGT

            nEstTotal+= nEst
            nGTTotal+= nGT
            nCorrectTotal += nCorrect


        stats = {
                "nGT": nGTTotal,
                "nEst": nEstTotal, 
                "nCorrect": nCorrectTotal,
                }

        return stats







    def computeStats(self, xBatch, notesBatch):
        # print(xBatch.shape)
        # print(len(notesBatch))
        nBatch = xBatch.shape[0]
        xBatch = xBatch.transpose(-1,-2)

        device = xBatch.device


        #[nBatch, nChannel, nSample]
        framesBatch = makeFrame(xBatch, self.hopSize, self.windowSize)

        # crf returned would be nBatch*nChannel flattened for batch processing
        crfBatch, ctxBatch = self.processFramesBatch(framesBatch)

        path = crfBatch.decode()

        # print(sum([len(p) for p in path]))
        intervalsBatch= []
        velocityBatch = []
        ofRefinedGTBatch = []
        for notes in notesBatch:
            data = prepareIntervals(notes, self.hopSize/self.fs, self.targetMIDIPitch)
            intervalsBatch .append(data["intervals"])
            velocityBatch.append(sum(data["velocity"], []))
            ofRefinedGTBatch.append(sum(data["endPointRefine"], []))

        intervalsBatch_flatten = sum(intervalsBatch , [])
        assert( len(intervalsBatch_flatten) == nBatch* len(self.targetMIDIPitch))



        # then compare intervals and path
        assert(len(path) == len(intervalsBatch_flatten))
            
        
        # print(sum([len(p) for p in intervalsBatch_flatten]), "intervalsGT")

        statsAll = [compareBracket(l1,l2) for l1, l2 in zip(path, intervalsBatch_flatten)]

        nGT = sum([_[0] for _ in statsAll])
        nEst = sum([_[1] for _ in statsAll])
        nCorrect= sum([_[2] for _ in statsAll])

        
        # omit pedal
        statsFramewiseAll = [compareFramewise(l1,l2) for l1, l2 in zip(path, intervalsBatch_flatten)]

        nGTFramewise = sum([_[0] for _ in statsFramewiseAll])
        nEstFramewise = sum([_[1] for _ in statsFramewiseAll])
        nCorrectFramewise = sum([_[2] for _ in statsFramewiseAll])




        # then make forced predictions about velocity and refined onset offset

        ctx_a_all, ctx_b_all, symIdx_all, scatterIdx_all = self.fetchIntervalFeaturesBatch(ctxBatch, intervalsBatch)

        attributeInput = torch.cat([ctx_a_all,
                                   ctx_b_all,
                                   ctx_a_all*ctx_b_all,
                                   ],dim = -1)



        logitsVelocity = self.velocityPredictor(attributeInput)
        pVelocity = F.softmax(logitsVelocity, dim = -1)

        
        #MSE
        w = torch.arange(128, device = device)
        velocity = (pVelocity*w).sum(-1)


        ofValue, _ = self.refinedOFPredictor(attributeInput).chunk(2, dim = -1)
        # ofValue = torch.sigmoid(ofValue)-0.5
        ofDist = torch.distributions.ContinuousBernoulli(logits=ofValue)

        ofValue = (ofDist.mean-0.5)/0.99
        ofValue = torch.clamp(ofValue, -0.5, 0.5)


        velocityBatch = sum(velocityBatch, [] )
        velocityBatch= torch.tensor(velocityBatch, dtype =torch.long, device = device)

        ofRefinedGTBatch = sum(ofRefinedGTBatch, [] )
        ofRefinedGTBatch = torch.tensor(ofRefinedGTBatch, device = device, dtype = torch.float)
        # compare p velocity with ofValue


        # ofValue-of
        seOF = (ofValue-ofRefinedGTBatch).pow(2).sum()
        seVelocity= (velocity-velocityBatch).pow(2).sum()

        # print(ofValue[0], ofRefinedGTBatch[0])
        # print(ofValue[-1], ofRefinedGTBatch[-1])

        stats = {
                "nGT": nGT,
                "nEst": nEst, 
                "nCorrect": nCorrect,
                "nGTFramewise": nGTFramewise,
                "nEstFramewise": nEstFramewise, 
                "nCorrectFramewise": nCorrectFramewise,
                "seVelocityForced": seVelocity.item(),
                "seOFForced": seOF.item(),
                }



        return stats

    def fetchIntervalFeaturesBatch(self, ctxBatch, intervalsBatch):
        # ctx: [N, SYM, T, D]
        ctx_a_all = []
        ctx_b_all = []
        symIdx_all = []
        scatterIdx_all = []
        device = ctxBatch.device
        T = ctxBatch.shape[-2]

        for idx, curIntervals in enumerate(intervalsBatch):
            nIntervals =len(sum(curIntervals, []))
            if nIntervals>0:
                symIdx = torch.tensor(listToIdx(curIntervals), dtype=torch.long, device = device)
                symIdx_all.append(symIdx)

                scatterIdx_all.append(idx*len(self.targetMIDIPitch)+ symIdx)

                indices = torch.tensor(sum(curIntervals, []), dtype =torch.long, device = device)
                # print(len(symIdx), len(indices[:,0]))

                ctx_a = ctxBatch[idx].flatten(0,1).index_select(dim = 0, index = indices[:, 0]+ symIdx*T)
                ctx_b = ctxBatch[idx].flatten(0,1).index_select(dim = 0, index = indices[:, 1]+ symIdx*T)

                ctx_a_all.append(ctx_a)
                ctx_b_all.append(ctx_b)

        ctx_a_all = torch.cat(ctx_a_all, dim = 0)
        ctx_b_all = torch.cat(ctx_b_all, dim = 0)
        symIdx_all= torch.cat(symIdx_all, dim = 0)
        scatterIdx_all= torch.cat(scatterIdx_all, dim = 0)

        return ctx_a_all, ctx_b_all, symIdx_all, scatterIdx_all




    def transcribeFrames(self,framesBatch, forcedStartPos= None, velocityCriteron = "hamming", onsetBound = None, lastFrameIdx = None):
        device = framesBatch.device
        nBatch=  framesBatch.shape[0]
        crfBatch, ctxBatch = self.processFramesBatch(framesBatch)
        nSymbols = len(self.targetMIDIPitch)
        nFrame = framesBatch.shape[-2]

        if lastFrameIdx is None:
            lastFrameIdx = nFrame-1



        path = crfBatch.decode(forcedStartPos = forcedStartPos, forward=False)

        assert(nSymbols*nBatch == len(path))

        # also get the last position for each path for forced decoding
        if onsetBound is not None:
            path = [[e for e in _ if e[0]<onsetBound] for _ in path]
        




        # then predict attributes associated with frames

        # obtain segment features

        nIntervalsAll =  sum([len(_) for _ in path])
        # print("#e:", nIntervalsAll)

        intervalsBatch = []
        for idx in range(nBatch):
            curIntervals =  path[idx*nSymbols: (idx+1)*nSymbols]
            intervalsBatch.append(curIntervals)
       
        if nIntervalsAll == 0:
            # nothing detected, return empty
            return [[] for _ in range(nBatch)], [0 for _ in range(len(path))]


        # then predict the attribute set

        ctx_a_all, ctx_b_all, symIdx_all, scatterIdx_all = self.fetchIntervalFeaturesBatch(ctxBatch, intervalsBatch)


        attributeInput = torch.cat([ctx_a_all,
                                   ctx_b_all,
                                   ctx_a_all*ctx_b_all,
                                   ],dim = -1)



        logitsVelocity = self.velocityPredictor(attributeInput)
        pVelocity = F.softmax(logitsVelocity, dim = -1)

        
        #MSE
        if velocityCriteron == "mse":
            w = torch.arange(128, device = device)
            velocity = (pVelocity*w).sum(-1)
        elif velocityCriteron == "match":
            #TODO: Minimal risk 
            # predict velocity, readout by minimizing the risk
            # 0.1 is usually the tolerance for the velocity, so....
            
            # It will never make so extreme predictions

            # create the risk matrix
            w = torch.arange(128, device = device)

            # [Predicted, Actual]

            tolerance = 0.1* 128
            utility = ((w.unsqueeze(1)- w.unsqueeze(0)).abs()<tolerance).float()

            r = pVelocity@utility

            velocity = torch.argmax(r, dim = -1)


        elif velocityCriteron == "hamming":
            # return the mode
            velocity = torch.argmax(pVelocity, dim = -1)

        elif velocityCriteron == "mae":
            # then this time return the median
            pCum = pVelocity.cumsum(-1)
            tmp = (pCum-0.5)>0
            w2 = torch.arange(128, 0. ,-1, device = device)

            velocity = torch.argmax(tmp*w2, dim = -1)


        else:
            raise Exception("Unrecognized criterion: {}".format(velocityCriteron))

        



        ofValue, ofPresence = self.refinedOFPredictor(attributeInput).chunk(2, dim = -1)
        # ofValue = torch.sigmoid(ofValue)-0.5
        ofDist = torch.distributions.ContinuousBernoulli(logits=ofValue)

        ofValue = (ofDist.mean-0.5)/0.99
        ofValue = torch.clamp(ofValue, -0.5, 0.5)

        ofPresence = ofPresence>0



        # print(velocity)
        # print(ofValue)

        # generate the final result


        # parse the list of path to (begin, end, midipitch, velocity) 


        velocity = velocity.cpu().detach().tolist()
        ofValue = ofValue.cpu().detach().tolist()
        ofPresence = ofPresence.cpu().detach().tolist()

        assert(len(velocity) == len(ofValue))
        assert(len(velocity) == nIntervalsAll)
         

        nCount = 0 

        notes = [[] for _ in range(nBatch)]

        frameDur = self.hopSize/self.fs

        # the last offset
        lastP = []

        for idx in range(nBatch):
            curIntervals = intervalsBatch[idx]


            for j, eventType in enumerate(self.targetMIDIPitch):
                lastEnd = 0
                curLastP = 0

                for k, aInterval in enumerate(curIntervals[j]):
                    # print(aInterval, eventType, velocity[nCount], ofValue[nCount])
                    isLast = (k == (len(curIntervals[j])-1) )
                    
                    curVelocity = velocity[nCount]

                    curOffset = ofValue[nCount]
                    start = (aInterval[0]+ curOffset[0] )*frameDur
                    end = (aInterval[1]+ curOffset[1])*frameDur

                    # ofPresence prediction is only used to distinguish the corner case that either onset or offset happens exactly on the first/last frame.

                    hasOnset = (aInterval[0]>0) or ofPresence[nCount][0]
                    hasOffset = (aInterval[1]<lastFrameIdx) or ofPresence[nCount][1]

                    assert(aInterval[0]>= 0)
                    # print(aInterval[0], aInterval[1], nFrame)
                    start = max(start, lastEnd)
                    end = max(end, start+1e-8)
                    lastEnd = end
                    curNote = Note(
                         start = start,
                         end = end,
                         pitch = eventType,
                         velocity = curVelocity,
                         hasOnset = hasOnset,
                         hasOffset = hasOffset)
                    
                    notes[idx].append(curNote)

                    if hasOffset:
                        curLastP = aInterval[1]
                    # if hasOnset and hasOffset:
                        # curLastP = aInterval[1]


                    nCount+= 1

                lastP.append(curLastP)

            notes[idx].sort(key = lambda x: (x.start, x.end,x.pitch))

        return notes, lastP



    def transcribe(self, x, stepInSecond = None, segmentSizeInSecond = None, discardSecondHalf=False, mergeIncompleteEvent = True):


        if stepInSecond is None and segmentSizeInSecond is None:
            stepInSecond = self.segmentHopSizeInSecond
            segmentSizeInSecond = self.segmentSizeInSecond

        x= x.transpose(-1,-2)

        # gain normalization
        # x = (x-x.mean())/(x.std()+1e-8)

        padTimeBegin = (segmentSizeInSecond-stepInSecond)

        x = F.pad(x, (math.ceil(padTimeBegin*self.fs), math.ceil(self.fs* (padTimeBegin))))

        nSample = x.shape[-1]
        

        eventsAll= []

        eventsByType= defaultdict(list)
        startFrameIdx = math.floor(padTimeBegin*self.fs/self.hopSize)
        startPos = [startFrameIdx]* len(self.targetMIDIPitch)
        # startPos =None

        stepSize = math.ceil(stepInSecond*self.fs/self.hopSize)*self.hopSize
        segmentSize = math.ceil(segmentSizeInSecond*self.fs)

        for i in range(0, nSample, stepSize):
            # t1 = time.time()

            j = min(i+ segmentSize, nSample)
            # print(i, j)


            beginTime = (i)/ self.fs -padTimeBegin
            # print(beginTime)

            curSlice = x[:, i:j]
            if curSlice.shape[-1]< segmentSize:
                # pad to the segmentSize
                curSlice = F.pad(curSlice, (0, segmentSize- curSlice.shape[-1]))


            curFrames = makeFrame(curSlice, self.hopSize, self.windowSize)

            lastFrameIdx = round(segmentSize/self.hopSize)
            # # print(curSlice.shape)
            # # print(startPos)
            # startPos = None
            if discardSecondHalf:
                onsetBound = stepSize
            else:
                onsetBound = None

            curEvents, lastP = self.transcribeFrames(curFrames.unsqueeze(0), forcedStartPos = startPos, velocityCriteron = "hamming", onsetBound= onsetBound, lastFrameIdx = lastFrameIdx)
            curEvents = curEvents[0]


            startPos = []
            for k in lastP:
                startPos.append(max(k-int(stepSize/self.hopSize), 0))

            # # shift all notes by beginTime
            for e in  curEvents:
                e.start += beginTime
                e.end  += beginTime 

                e.start = max(e.start, 0)
                e.end = max(e.end, e.start)
                # print(e.start, e.end, e.pitch, e.hasOnset, e.hasOffset)


            for e in curEvents:
                if mergeIncompleteEvent:
                    if len(eventsByType[e.pitch])>0:
                        last_e = eventsByType[e.pitch][-1]

                        # test if e overlap with the last event 
                        if e.start < last_e.end:


                            if e.hasOnset: #and e.hasOffset:
                                eventsByType[e.pitch][-1] = e
                            else:
                                # merge two events
                                eventsByType[e.pitch][-1].hasOffset = e.hasOffset
                                eventsByType[e.pitch][-1].end = max(e.end, last_e.end)
                                # eventsByType[e.pitch][-1].end = max(e.end, last_e.end)

                            continue


                if e.hasOnset:
                    eventsByType[e.pitch].append(e)
            


            eventsAll.extend(curEvents)

        # handling incomplete events in the last segment
        for eventType in eventsByType:
            if len(eventsByType[eventType])>0:
                eventsByType[eventType][-1].hasOffset = True

        # flatten all events
        eventsAll = sum(eventsByType.values(), [])



        # post filtering
        eventsAll = [n for n in eventsAll if n.hasOffset]

        eventsAll = resolveOverlapping(eventsAll)




        return eventsAll



if __name__ == "__main__":
    device = "cuda"
    model = TransKun(Config()).to(device)
    print("#Param(M):", computeParamSize(model))
    # frames = torch.randn(4, 2, 432, 4096).to(device)

    
    # for i in range(1000):
        # model.processFramesBatch(frames)
    
    datasetPath = "/data/maestro/maestro-v2.0.0/"
    datasetPicklePath = "data_v3/val.pickle"

    dataset = DatasetMaestro(datasetPath, datasetPicklePath)

    dataIter = DatasetMaestroIterator(dataset, 5, 10, notesStrictlyContained=False)

    batchSize = 3
    dataloader= torch.utils.data.DataLoader(dataIter, batch_size = batchSize, collate_fn = collate_fn , num_workers=0, shuffle=True)

    for i in range(1000):
        for j, batch in enumerate(dataloader):
            model.zero_grad()
            notesBatch = [sample["notes"] for sample in batch]
            audioSlices = torch.stack(
                    [torch.from_numpy(sample["audioSlice"]) for sample in batch], dim = 0) . to(device)

            logp = model.log_prob(audioSlices, notesBatch)

            logp.sum().backward()


