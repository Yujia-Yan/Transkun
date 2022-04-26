
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .Util import *
import torch.distributed as dist


class ConvBlock_ablation(nn.Module):
    def __init__(
            self,
            inputSize,
            outputSize,
            hiddenSize,
            kernelSize,
            stride = 1,
            dropoutProb = 0.0
            ):
        super().__init__()
        if dist.is_available() and dist.is_initialized():
            print("syncBatchnorm enabled")
            from .SyncBN import SynchronizedBatchNorm2d
            BatchNorm2d = SynchronizedBatchNorm2d
        else:
            BatchNorm2d = nn.BatchNorm2d

        if isinstance(stride,int):
            stride = (stride, stride)

        nPad = (kernelSize-1, kernelSize-1) 

        self.conv1 = nn.Conv2d(inputSize, hiddenSize, kernelSize, padding =  kernelSize//2)


        self.bn1 = BatchNorm2d(hiddenSize, momentum = 0.01 )

        self.conv2 = nn.Conv2d(hiddenSize, outputSize, kernelSize, padding = kernelSize//2)
        self.bn2 = BatchNorm2d(outputSize, momentum = 0.01)


        self.downSampler = nn.Identity()

        if stride != (1,1):
            self.downSampler = nn.AvgPool2d(stride, stride=stride)


        self.dropout = nn.Dropout(dropoutProb)



        
        


    def forward(self, x):

         

        z = self.conv1(x.view_as(x))
        z = self.bn1(z)
        z = F.gelu(z)

        z = self.conv2(z.view_as(z))
        z = self.bn2(z)
        z = F.gelu(z)

        z = self.downSampler(z)

        return z

class SimpleRNN(nn.Module):
    
    def __init__(self, inputSize, hiddenSize, outputSize,  nLayers, dropoutProb):
        super().__init__()

        self.grus = nn.GRU(inputSize, hiddenSize, num_layers = nLayers, bidirectional=True, dropout=dropoutProb)
        self.outProj = nn.Linear(hiddenSize*2, outputSize)


    def forward(self,x ):
        if self.training:
            checkpoint = torch.utils.checkpoint.checkpoint
        else:
            checkpoint = checkpointByPass

        y, _ = checkpoint(self.grus,x)
        y = self.outProj(y)

        return y


class ScoreMatrixPostProcessor(nn.Module):
    def __init__(self, nTarget, nHidden, dropoutProb):
        super().__init__()

        self.map = nn.Sequential(
                nn.Conv2d(nTarget, nHidden, 3, padding= 2),
                nn.GELU(),
                nn.Dropout(dropoutProb),
                nn.Conv2d(nHidden, nTarget, 3)
                )


    def forward(self, S):
        if self.training:
            checkpointSequential = torch.utils.checkpoint.checkpoint_sequential
        else:
            checkpointSequential = checkpointSequentialByPass
        S = S.permute(2, 3, 0,1)
        S = checkpointSequential(self.map,2, S)
        S = S.permute(2,3, 0, 1).contiguous()
        return S

class PairwiseFeatureBatch(nn.Module):
    def __init__(self,
                 inputSize,
                 outputSize,
                 dropoutProb = 0.0,
                 lengthScaling=True,
                 postConv=True,
                 disableUnitary=False,
                 hiddenSize = None
                 ):
        super().__init__()
    

        if hiddenSize is None:
            hiddenSize = outputSize*4
        self.scoreMap = nn.Sequential(
                nn.Linear(inputSize*6, hiddenSize),
                nn.GELU(),
                nn.Dropout(dropoutProb),
                nn.Linear(hiddenSize, hiddenSize),
                nn.GELU(),
                nn.Dropout(dropoutProb),
                nn.Linear(hiddenSize, outputSize),
                )

        self.scoreMapSkip = nn.Sequential(
                nn.Linear(inputSize*3, hiddenSize),
                nn.GELU(),
                nn.Dropout(dropoutProb),
                nn.Linear(hiddenSize, hiddenSize),
                nn.GELU(),
                nn.Dropout(dropoutProb),
                nn.Linear(hiddenSize, outputSize)
                )

        self.lengthScaling = lengthScaling

        self.disableUnitary=disableUnitary
        
        self.post = nn.Identity()
        if postConv:
            self.post = ScoreMatrixPostProcessor(outputSize, outputSize*3, dropoutProb)

    def computeChunk(self, x, x_cum, x_sqr_cum,x_cube_cum, idxA, idxB):
        # A: end
        # B: begin
        curA = x[idxA]
        curB = x[idxB]

        lengthBA = (idxA-idxB)+1
        lengthBA = lengthBA.view(-1,1, 1)

        moment1 = (x_cum[idxA+1]- x_cum[idxB])/lengthBA
        moment2 = (x_sqr_cum[idxA+1]- x_sqr_cum[idxB])/lengthBA
        moment3 = (x_cube_cum[idxA+1]- x_cube_cum[idxB])/lengthBA


    

        curInput = torch.cat([curA, curB,  curA*curB, moment1, moment2, moment3], dim = -1)
        curScore = self.scoreMap(curInput)


        return curScore

    def computeSkipScore(self, x):
        curA = x[:-1]
        curB = x[1:]
        curInput = torch.cat([curA, curB,  curA*curB], dim = -1)
        curScore = self.scoreMapSkip(curInput)
        return curScore

    def forward(self, x, nBlock = 4000):
        if self.training:
            checkpoint = torch.utils.checkpoint.checkpoint
        else:
            checkpoint = checkpointByPass

        # input shape: [T, nBatch, .]
        assert(len(x.shape)==3)
        nEntry = x.shape[0]
        indices = torch.tril_indices(nEntry,nEntry, device = x.device)

        nTotal = indices.shape[1]

        S_all = []

        x_cum = torch.cumsum(F.pad(x, (0,0,0,0,1,0)), dim =0)
        x_sqr_cum = torch.cumsum(F.pad(x.pow(2), (0,0,0,0,1,0)), dim =0)
        x_cube_cum = torch.cumsum(F.pad(x.pow(3), (0,0,0,0,1,0)), dim =0)

        for lIdx in range(0, nTotal, nBlock):
            if lIdx+nBlock< nTotal:
                idxA = indices[0, lIdx:lIdx+nBlock]
                idxB = indices[1, lIdx:lIdx+nBlock]
            else:
                idxA = indices[0, lIdx:]
                idxB = indices[1, lIdx:]

            # curScore = self.computeChunk(x, idxA, idxB)
            curScore = checkpoint(self.computeChunk, x, x_cum, x_sqr_cum, x_cube_cum, idxA, idxB)

            S_all.append(curScore)

        s_val = torch.cat(S_all, dim = 0)

        S_coo = torch.sparse_coo_tensor(indices, s_val, (nEntry, nEntry, s_val.shape[-2], s_val.shape[-1]))

        S = S_coo.to_dense()
        # print(S.std(), S.max(), S.min())

        S =  self.post(S)
        # print(S.std(), S.max(), S.min())


        if self.lengthScaling:
            tmpIdx = torch.arange(nEntry, device = S.device)
            lenBA = (tmpIdx.unsqueeze(-1)- tmpIdx.unsqueeze(0)).abs().clamp(1)
            S = lenBA.unsqueeze(-1).unsqueeze(-1)*S
            # curScore = lengthBA*curScore
        S_skip = self.computeSkipScore(x)
        if self.disableUnitary:
            S_skip =S_skip*0


        return S, S_skip

