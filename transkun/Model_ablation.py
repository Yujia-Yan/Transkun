import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint
import math
from .Data import *
from .Util import *
from .Layers_ablation import *
import torch.distributed as dist
from .Evaluation import *


from . import CRF


class ModelConfig:
    def __init__(self):

        self.f_min = 30
        self.f_max = 8000
        self.n_mels = 229

        self.hopSize = 1024
        self.windowSize =4096
        self.fs = 44100
        self.nExtraWins = 5



        self.preConvSpec = [
                {"outputSize": 48, "hiddenSize": 48, "kernelSize":3, "stride":(1,2), "dropoutProb":0.0},
                {"outputSize": 64, "hiddenSize": 64, "kernelSize":3, "stride":(1,2), "dropoutProb":0.0},
                {"outputSize": 92, "hiddenSize": 92, "kernelSize":3, "stride":(1,2), "dropoutProb": 0.0},
                {"outputSize": 128, "hiddenSize": 128, "kernelSize":3, "stride":(1,2), "dropoutProb":0.0},
                ]

        # context layers
        self.ctxSize = 512
        self.nLayersCtx = 2
        self.rnnHiddenSize = 256

         
        
        self.lengthScaling = True
        self.postConv = True
        self.disableUnitary = False





        self.pitchEmbedSize = 256

        self.scoreDropoutProb = 0.1
        self.contextDropoutProb = 0.1

        self.velocityDropoutProb = 0.1
        self.refinedOFDropoutProb= 0.1

    def __repr__(self):
        return repr(self.__dict__)

Config = ModelConfig

class PreLayer(nn.Module):
    def __init__(
            self, 
            inputSize,
            nEntry,
            spec 
            ):
        super().__init__()
        layers = []

        curInputSize = inputSize
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))

        testVec = self.dummy.new_zeros(1, inputSize, 1, nEntry)
        for s in spec:
            curLayer = ConvBlock_ablation(curInputSize, **s)
            layers.append(
                    curLayer
                )
            curInputSize = s["outputSize"]
            testVec = curLayer.conv1(testVec)
            testVec = curLayer.conv2(testVec)
            testVec = curLayer.downSampler(testVec)

        self.layers = nn.ModuleList(layers)


        self.inputSize =inputSize
        self.outputSize = curInputSize
        self.nEntry = nEntry
        self.nEntryOut = testVec.shape[-1]


        # testing to get the output dimension
        


    def forward(self, x):
        # a fix for torch.utils.checkpoint.checkpoint 
        z = x+ 0*self.dummy


        if self.training:
            checkpoint = torch.utils.checkpoint.checkpoint
        else:
            checkpoint = checkpointByPass

        for l in self.layers:
            z = checkpoint(l,z)

        
        return z


class TransKun(torch.nn.Module):
    Config = ModelConfig
    def __init__(self, conf):
        super().__init__()

        self.conf = conf
        self.hopSize = conf.hopSize

        self.windowSize = conf.windowSize
        self.fs = conf.fs




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

        lengthScaling = True
        if hasattr(conf, "lengthScaling"):
            lengthScaling = conf.lengthScaling


        disableUnitary = False
        if hasattr(conf, "disableUnitary"):
            disableUnitary = conf.disableUnitary

        self.pairwiseScore = PairwiseFeatureBatch(conf.ctxSize, len(self.targetMIDIPitch), dropoutProb = conf.scoreDropoutProb, lengthScaling=lengthScaling, postConv = conf.postConv, disableUnitary = disableUnitary)

        self.pitchEmbedding = nn.Embedding(len(self.targetMIDIPitch), conf.pitchEmbedSize)

        self.velocityPredictor = nn.Sequential(
                                nn.Linear(conf.ctxSize*3 + conf.pitchEmbedSize, 512),
                                nn.GELU(),
                                nn.Dropout(conf.velocityDropoutProb),
                                nn.Linear(512, 512),
                                nn.GELU(),
                                nn.Dropout(conf.velocityDropoutProb),
                                nn.Linear(512, 128)
                                )


        self.refinedOFPredictor = nn.Sequential(
                                nn.Linear(conf.ctxSize*3+conf.pitchEmbedSize, 512),
                                nn.GELU(),
                                nn.Dropout(conf.refinedOFDropoutProb),
                                nn.Linear(512, 128),
                                nn.GELU(),
                                nn.Dropout(conf.refinedOFDropoutProb),
                                nn.Linear(128, 2)
                                )

        
        self.preLayer = PreLayer(inputSize = 1*self.framewiseFeatureExtractor.nChannel, 
                nEntry = self.framewiseFeatureExtractor.outputDim,
                                 spec = conf.preConvSpec)

        
        if dist.is_available() and dist.is_initialized():
            from .SyncBN import SynchronizedBatchNorm2d
            BatchNorm2d = SynchronizedBatchNorm2d
        else:
            BatchNorm2d = nn.BatchNorm2d

        self.inputProj = nn.Sequential(
                nn.Linear(self.preLayer.outputSize*self.preLayer.nEntryOut, conf.ctxSize)
                )


        self.contextModel = SimpleRNN(conf.ctxSize, hiddenSize = conf.rnnHiddenSize, outputSize = conf.ctxSize, nLayers = conf.nLayersCtx, dropoutProb= conf.contextDropoutProb)


        
    def getDevice(self):
        return self.pitchEmbedding.weight.device

                

    def processFramesBatch(self, framesBatch):
        
        nBatch = framesBatch.shape[0]
        # nChannel = framesBatch.shape[1]
        
        # gain normalization
        framesBatchMean = torch.mean(framesBatch, dim = [1,2,3], keepdim=True)
        framesBatchStd = torch.std(framesBatch, dim = [1,2,3], keepdim=True)
        framesBatch = (framesBatch - framesBatchMean)/(framesBatchStd+ 1e-8)

        featuresBatch = self.framewiseFeatureExtractor(framesBatch).contiguous()

        # now with shape [nBatch, nAudioChannel, nStep, NFreq, nChannel]
        nChannel = 1


        featuresBatch  = featuresBatch.view(nBatch*nChannel, *featuresBatch.shape[-3:])
        # now with shape [nBatch* nAudioChannel, nStep, NFreq, nChannel]
        featuresBatch = featuresBatch.permute(0, 3,1,2)
        # now with shape [nBatch* nAudioChannel, nChannel, nStep, NFreq]

        # print(featuresBatch.shape)
        # now conv layers
        featuresBatch = self.preLayer(featuresBatch.view_as(featuresBatch))

        featuresBatch = featuresBatch.view(nBatch,nChannel, *featuresBatch.shape[-3:])

    
        featuresBatch = featuresBatch[:, 0, ...]
        # now with shape [nBatch,  nFeature, nStep, nFreq)

    
        # then it goes a context model
        featuresBatch = featuresBatch.transpose(-2,-3).flatten(-2,-1)



        ctx = self.inputProj(featuresBatch)
        ctx= ctx.permute(1, 0, 2)

        ctx = self.contextModel(ctx)

        # ctx = self.contextModel2(ctx)

        # then create pairwise scores

        # now with shape [ nStep, nStep, nBatch, nFeat] 
        # print(ctx.shape)
        S_batch, S_skip_batch = self.pairwiseScore(ctx)




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
        for notes in notesBatch:
            data = prepareIntervals(notes, self.hopSize/self.fs, self.targetMIDIPitch)
            intervalsBatch .append(data["intervals"])
            velocityBatch.append(sum(data["velocity"], []))
            ofRefinedGTBatch.append(sum(data["endPointRefine"], []))


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

            pitchEmbed_all = self.pitchEmbedding(symIdx_all)

            attributeInput = torch.cat([ctx_a_all,
                                       ctx_b_all,
                                       ctx_a_all*ctx_b_all,
                                       pitchEmbed_all],dim = -1)

            
            # prepare groundtruth for velocity

            velocityBatch = sum(velocityBatch, [] )
            ofRefinedGTBatch = sum(ofRefinedGTBatch, [] )

            logitsVelocity = self.velocityPredictor(attributeInput)
            logitsVelocity = F.log_softmax(logitsVelocity, dim = -1)

            velocityBatch= torch.tensor(velocityBatch, dtype =torch.long, device = device)

            logProbVelocity = torch.gather(logitsVelocity, dim = -1, index = velocityBatch.unsqueeze(-1)).squeeze(-1)

            ofRefinedGTBatch = torch.tensor(ofRefinedGTBatch, device = device, dtype = torch.float)
            

            # shift it to [0,1]
            # print("GT:", ofRefinedGTBatch)

            ofRefinedGTBatch = ofRefinedGTBatch*0.99+0.5

            ofValue = self.refinedOFPredictor(attributeInput)
            # ofValue = F.logsigmoid(ofValue)

            ofDist = torch.distributions.ContinuousBernoulli(logits=ofValue)

            logProbOF = ofDist.log_prob(ofRefinedGTBatch).sum(-1)




            # scatter them back
            logProb = logProb.view(-1)

            
            logProb = logProb.scatter_add(-1, scatterIdx_all, logProbVelocity+ logProbOF)
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
            nGT = len(gt)

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
        pitchEmbed_all = self.pitchEmbedding(symIdx_all)

        attributeInput = torch.cat([ctx_a_all,
                                   ctx_b_all,
                                   ctx_a_all*ctx_b_all,
                                   pitchEmbed_all],dim = -1)



        logitsVelocity = self.velocityPredictor(attributeInput)
        pVelocity = F.softmax(logitsVelocity, dim = -1)

        
        #MSE
        w = torch.arange(128, device = device)
        velocity = (pVelocity*w).sum(-1)


        ofValue = self.refinedOFPredictor(attributeInput)
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
        ctx_a_all = []
        ctx_b_all = []
        symIdx_all = []
        scatterIdx_all = []
        device = ctxBatch.device
        for idx, curIntervals in enumerate(intervalsBatch):
            nIntervals =len(sum(curIntervals, []))
            if nIntervals>0:
                symIdx = torch.tensor(listToIdx(curIntervals), dtype=torch.long, device = device)
                symIdx_all.append(symIdx)

                scatterIdx_all.append(idx*len(self.targetMIDIPitch)+ symIdx)

                indices = torch.tensor(sum(curIntervals, []), dtype =torch.long, device = device)

                ctx_a = ctxBatch[indices[:, 0], idx]
                ctx_b = ctxBatch[indices[:, 1], idx]
                ctx_a_all.append(ctx_a)
                ctx_b_all.append(ctx_b)

        ctx_a_all = torch.cat(ctx_a_all, dim = 0)
        ctx_b_all = torch.cat(ctx_b_all, dim = 0)
        symIdx_all= torch.cat(symIdx_all, dim = 0)
        scatterIdx_all= torch.cat(scatterIdx_all, dim = 0)

        return ctx_a_all, ctx_b_all, symIdx_all, scatterIdx_all




    def transcribeFrames(self,framesBatch, forcedStartPos= None, velocityCriteron = "hamming", onsetBound = None):
        device = framesBatch.device
        nBatch=  framesBatch.shape[0]
        crfBatch, ctxBatch = self.processFramesBatch(framesBatch)
        nSymbols = len(self.targetMIDIPitch)




        path = crfBatch.decode(forcedStartPos = forcedStartPos, forward=False)

        assert(nSymbols*nBatch == len(path))

        # also get the last position for each path for forced decoding
        # endPos = 
        if onsetBound is not None:
            path = [[e for e in _ if e[0]<onsetBound] for _ in path]
        
        lastP = []

        for curP in path:
            if len(curP) == 0:
                lastP.append(0)
            else:
                lastP.append(curP[-1][1])




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
            return [[] for _ in range(nBatch)],lastP


        # then predict the attribute set

        ctx_a_all, ctx_b_all, symIdx_all, scatterIdx_all = self.fetchIntervalFeaturesBatch(ctxBatch, intervalsBatch)

        pitchEmbed_all = self.pitchEmbedding(symIdx_all)

        attributeInput = torch.cat([ctx_a_all,
                                   ctx_b_all,
                                   ctx_a_all*ctx_b_all,
                                   pitchEmbed_all],dim = -1)



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

        



        ofValue = self.refinedOFPredictor(attributeInput)
        # ofValue = torch.sigmoid(ofValue)-0.5
        ofDist = torch.distributions.ContinuousBernoulli(logits=ofValue)

        ofValue = (ofDist.mean-0.5)/0.99
        ofValue = torch.clamp(ofValue, -0.5, 0.5)

        # print(velocity)
        # print(ofValue)

        # generate the final result


        # parse the list of path to (begin, end, midipitch, velocity) 


        velocity = velocity.cpu().detach().tolist()
        ofValue = ofValue.cpu().detach().tolist()

        assert(len(velocity) == len(ofValue))
        assert(len(velocity) == nIntervalsAll)
         

        nCount = 0 

        notes = [[] for _ in range(nBatch)]

        frameDur = self.hopSize/self.fs


        for idx in range(nBatch):
            curIntervals = intervalsBatch[idx]

            for j, eventType in enumerate(self.targetMIDIPitch):
                lastEnd = 0
                for aInterval in curIntervals[j]:
                    # print(aInterval, eventType, velocity[nCount], ofValue[nCount])
                    
                    curVelocity = velocity[nCount]

                    curOffset = ofValue[nCount]
                    start = (aInterval[0]+ curOffset[0] )*frameDur
                    end = (aInterval[1]+ curOffset[1])*frameDur

                    start = max(start, lastEnd)
                    end = max(end, start+1e-8)
                    lastEnd = end
                    curNote = Note(
                         start = start,
                         end = end,
                         pitch = eventType,
                         velocity = curVelocity)
                    
                    notes[idx].append(curNote)

                    nCount+= 1


            notes[idx].sort(key = lambda x: (x.start, x.end,x.pitch))

        # sort all 
        # print(notes)

        return notes, lastP



    def transcribe(self, x, stepInSecond = 10, segmentSizeInSecond = 20, discardSecondHalf=False):



        x= x.transpose(-1,-2)
        padTimeBegin = (segmentSizeInSecond-stepInSecond)

        x = F.pad(x, (math.ceil(padTimeBegin*self.fs), math.ceil(self.fs* (padTimeBegin))))

        nSample = x.shape[-1]
        

        eventsAll= []

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

            # curSlice = frames[:, i:j, :]
            curSlice = x[:, i:j]
            curFrames = makeFrame(curSlice, self.hopSize, self.windowSize)

            # # print(curSlice.shape)
            # # print(startPos)
            # startPos = None
            if discardSecondHalf:
                onsetBound = stepSize
            else:
                onsetBound = None

            curEvents, lastP = self.transcribeFrames(curFrames.unsqueeze(0), forcedStartPos = startPos, velocityCriteron = "hamming", onsetBound= onsetBound)
            curEvents = curEvents[0]


            startPos = []
            for k in lastP:
                startPos.append(max(k-int(stepSize/self.hopSize), 0))

            # # shift all notes by beginTime
            for e in  curEvents:
                e.start += beginTime
                e.end  += beginTime 

                e.start = max(e.start, 0)
                e.end = max(e.end, e.start+1e-5)




            # t2 = time.time()
            # # print("elapsed:", t2-t1)

            eventsAll.extend(curEvents)


        # check overlapping of eventsAll after adding the refined position

        eventsAll = resolveOverlapping(eventsAll)




        return eventsAll


