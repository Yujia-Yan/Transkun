import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List,Tuple, Optional

"""
    The neural semiCRF module for handling multiple tracks of non-overlapping intervalsc
    Author: Yujia Yan
"""

     
@torch.jit.script
def viterbiBackward(score, noiseScore, forcedStartPos: Optional[List[int]]=None):
    # score: [nEndPos,  nBeginPos, nBatch]
    # noiseScore: [nEndPos-1, nBatch]

    assert(len(score.shape) == 3)
    assert(score.shape[0] == score.shape[1])
    T = score.shape[0]
    nBatch = score.shape[2]
    
    q = torch.zeros(T, nBatch, device = score.device)

    # for back tracking
    ptr = []

    scoreT = score.transpose(0,1).contiguous()

    q[T-1] = score[T-1, T-1, :]* (score[T-1,T-1,:]>0)

    for i in range(1,T):

        # update v_on
        subScore  = scoreT[T-i-1, T-i:,:]

        tmp = torch.cat(
            [
            q[T-i:T-i+1, :] + noiseScore[T-i-1, :],              # skip
            q[T-i:, :]+ subScore       # an interval 
            ],
            dim = 0
        )
        
        curV, selection = tmp.max(dim = 0)

        ptr.append(selection-1)


        singletonMask = score[T-i-1, T-i-1,:]>0

        q[T-i-1] = curV+ score[T-i-1,T-i-1,:]*singletonMask


    qFinal = q[0]

    ptr= torch.stack(ptr, dim = 0).cpu()

    scoreDiagInclusion = (torch.diagonal(score, dim1= 0, dim2=1)>0).cpu()


    if forcedStartPos is None:
        forcedStartPos = [0]* nBatch


    # perform backtracking 
    result: List[List[Tuple[int, int]]]  =  []
    # print(ptr)

    
    for idx in range(nBatch):
        j = forcedStartPos[idx]

        curResult : List[Tuple[int, int]]  = []


        curDiag = scoreDiagInclusion[idx]
        while j< T-1:
            # print(j)
            curSelecton= int(ptr[T-j-2][idx])

            if bool(curDiag[j]):
                curResult.append((j,j))


            if curSelecton<0:
                j += 1
            else:
                # print("curSelect:", curSelecton)
                i = curSelecton+j+1

                # print((i,j))
                curResult.append((j,i))

                j = i
        

        if score[T-1,T-1, idx]>0:
            curResult.append((T-1,T-1))
        
        

        result.append(curResult)

    return result
     
@torch.jit.script
def viterbi(score, noiseScore, forcedStartPos: Optional[List[int]]=None):
    # score: [nEndPos,  nBeginPos, nBatch]
    # noiseScore: [nEndPos-1, nBatch]

    assert(len(score.shape) == 3)
    assert(score.shape[0] == score.shape[1])
    T = score.shape[0]
    nBatch = score.shape[2]
    
    v = torch.zeros(T, nBatch, device = score.device)

    # for back tracking
    ptr = []


    v[0] = score[0,0, :]* (score[0,0,:]>0)

    for i in range(1,T):

        # update v_on
        subScore = score[i, :i, :]

        tmp = torch.cat(
            [
            v[i-1:i,:] + noiseScore[i-1, :],              # skip
            v[:i, :]+ subScore       # an interval 
            ],
            dim = 0
        )
        
        curV, selection = tmp.max(dim = 0)

        ptr.append(selection-1)


        singletonMask = score[i, i,:]>0

        v[i] = curV+ score[i,i,:]*singletonMask


    vFinal = v[-1]
    # print(vFinal)

    ptr= torch.stack(ptr, dim = 0).cpu()

    scoreDiagInclusion = (torch.diagonal(score, dim1= 0, dim2=1)>0).cpu()




    # perform backtracking 
    result: List[List[Tuple[int, int]]]  =  []
    # print(ptr)

    if forcedStartPos is None:
        forcedStartPos = [T-1]* nBatch
    
    for idx in range(nBatch):
        j = forcedStartPos[idx]
        # c = int(cFinal[idx] )

        curResult : List[Tuple[int, int]]  = []

        curDiag = scoreDiagInclusion[idx]

        while j>0:
            # print(j)
            # if c == 0:
            curSelecton= int(ptr[j-1][idx])

            if bool(curDiag[j]):
                curResult.append((j,j))


            if curSelecton<0:
                # skip
                j = j-1
            else:
                i =curSelecton 

                # print((i,j))
                curResult.append((i,j))

                j = i
        
        if score[0,0, idx]>0:
            curResult.append((0,0))
        
        # # flip the list
        curResult.reverse()


        result.append(curResult)

    
    return result



@torch.jit.script
def computeLogZ(score, noiseScore):
    # [nEndPos,  nBeginPos, nBatch]
    assert(len(score.shape) == 3)
    assert(len(noiseScore.shape) == 2)
    assert(score.shape[0] == score.shape[1])
    T = score.shape[0]
    nBatch = score.shape[2]

    assert(noiseScore.shape[0] == T-1)


    v = F.softplus(score[0,0,:]).unsqueeze(0)

    for i in range(1, T):
        subScore = score[i, :i, :]


        tmp = torch.cat(
            [
            v[i-1:i,:] + noiseScore[i-1,:],              # skip
            v[:i, :]+ subScore       # an interval 
            ],
            dim = 0
        )

        curV= tmp.logsumexp(dim = 0) + F.softplus(score[i, i,:])

        v = torch.cat( [ v, curV.unsqueeze(0)], dim = 0)
        # curV= tmp.logsumexp(dim = 0) + singleScoreSP[i,:]#F.softplus(score[i, i,:])
        # curV = torch.logaddexp(
                # v[i-1,:] + noiseScore[i-1,:],              # skip
                # torch.logsumexp(v[:i, :]+subScore, dim = 0) )
        # curV = curV+singleScoreSP[i,:]
        
        # v = torch.cat( [ v, curV.unsqueeze(0)], dim = 0)
    
    vFinal = v[-1]
    

    return vFinal

# the autodiff backward is too slow, let write our own
@torch.jit.script
def forward_backwardOld(score, noiseScore):
    # [nEndPos,  nBeginPos, nBatch]
    assert(len(score.shape) == 3)
    assert(len(noiseScore.shape) == 2)
    assert(score.shape[0] == score.shape[1])
    T = score.shape[0]
    nBatch = score.shape[2]
    assert(noiseScore.shape[0] == T-1)


    # forward pass
    singleScoreSP = F.softplus(torch.diagonal(score, dim1=0, dim2=1)).transpose(-1,-2)
    # v = F.softplus(score[0,0,:]).unsqueeze(0)

    v = score.new_zeros(T, nBatch)
    v[0] = singleScoreSP[0, :]
    # v = singleScoreSP[0, :].unsqueeze(0)

    for i in range(1, T):
        subScore = score[i, :i, :]


        # tmp = torch.cat(
            # [
            # v[i-1:i,:] + noiseScore[i-1,:],              # skip
            # v[:i, :]+ subScore       # an interval 
            # ],
            # dim = 0
        # )

        # curV= tmp.logsumexp(dim = 0) + singleScoreSP[i,:]#F.softplus(score[i, i,:])
        # curV = torch.logaddexp(
                # v[i-1,:] + noiseScore[i-1,:],              # skip
                # torch.logsumexp(v[:i, :]+subScore, dim = 0) )
        # curV = curV+singleScoreSP[i,:]
        
        # v = torch.cat( [ v, curV.unsqueeze(0)], dim = 0)

        v[i] = torch.logaddexp(
                v[i-1,:] + noiseScore[i-1,:],              # skip
                torch.logsumexp(v[:i, :]+subScore, dim = 0) )
        v[i] += singleScoreSP[i,:]
        # v[i] = curV

    logZ = v[-1]



    # then do a backward pass
    scoreT = score.transpose(0,1).contiguous()
    # print("backward")

    # q = F.softplus(score[T-1, T-1, :]).unsqueeze(0)
    q = score.new_zeros(T, nBatch)
    q[T-1] = singleScoreSP[T-1, :]
    # q = singleScoreSP[T-1, :].unsqueeze(0)
    for i in range(1, T):
        subScore  = scoreT[T-i-1, T-i:,:]
        # tmp = torch.cat(
            # [
            # q[0:1, :] + noiseScore[T-i-1,:],              # skip
            # q + subScore       # an interval 
            # ],
            # dim = 0
        # )
        # curQ = tmp.logsumexp(dim =0) + singleScoreSP[T-i-1,:]#F.softplus(scoreT[T-i-1, T-i-1, :])
        # curQ= torch.logaddexp(
            # q[0, :] + noiseScore[T-i-1,:],              # skip
            # torch.logsumexp(q + subScore, dim=0)       # an interval 
            # )
        # curQ = curQ + singleScoreSP[T-i-1,:]

        # q = torch.cat([curQ.unsqueeze(0), q], dim = 0)

        q[T-i-1] = torch.logaddexp(
            q[T-i, :] + noiseScore[T-i-1,:],              # skip
            torch.logsumexp(q[T-i:] + subScore, dim=0)       # an interval 
            ) + singleScoreSP[T-i-1,:]

        # q = torch.cat([curQ.unsqueeze(0), q], dim = 0)


    # print("v:", v[-1])
    # print("q", q[0])
    # print(v[-1]-q[0])
    # grad except for the diagonal can be computed in the following way
    # grad = (v.unsqueeze(0)+ q.unsqueeze(1) + score -logZ ).exp()

    # diagonal elements are computed in this way
    # grad = (v.unsqueeze(0)+ q.unsqueeze(1) + score - 2* F.softplus(score) -logZ ).exp()

    grad = (v.unsqueeze(0)+ q.unsqueeze(1)-logZ + score)#.exp()

    # minus 2* F.softplus(diag of score)
    grad = grad - 2*torch.diag_embed(F.softplus(torch.diagonal(score, dim1 = 0, dim2 = 1)), dim1= 0, dim2 = 1)

    # grad = grad#*torch.ones(T, T, device= grad.device).tril().unsqueeze(-1)


     
    grad = grad*torch.ones(T, T, device= grad.device).tril().unsqueeze(-1)

    grad = grad.exp()

    grad = grad*torch.ones(T, T, device= grad.device).tril().unsqueeze(-1)

    # for grad of noise, it is similar

    
    gradNoise = (v[:-1]+ q[1:] + noiseScore-logZ)
    # gradNoise = gradNoise - gradNoise*(gradNoise>0)
    gradNoise = gradNoise.exp()

    
    # zeroout all the upper triangular part
    # print(grad)
    # print(grad[:,:,0])
    # print("grad:", grad[:,:,0])
    # print("gradNoise:",gradNoise[:,0])


    return logZ, grad, gradNoise


@torch.jit.script
def forward_backward(score, noiseScore):
    # [nEndPos,  nBeginPos, nBatch]
    assert(len(score.shape) == 3)
    assert(len(noiseScore.shape) == 2)
    assert(score.shape[0] == score.shape[1])
    T = score.shape[0]
    nBatch = score.shape[2]
    assert(noiseScore.shape[0] == T-1)
    

    # make score symmetric
    scoreFlip = torch.flip(score, dims = [0,1]).transpose(0,1)
    noiseScoreFlip = torch.flip(noiseScore, (0,))


    scoreFB= torch.cat([score, scoreFlip], dim = -1)
    noiseScoreFB = torch.cat([noiseScore, noiseScoreFlip], dim = -1)

    
    # forward pass
    singleScoreSP = F.softplus(torch.diagonal(scoreFB, dim1=0, dim2=1)).transpose(-1,-2)
    # v = F.softplus(score[0,0,:]).unsqueeze(0)

    v = score.new_zeros(T, nBatch*2)
    v[0] = singleScoreSP[0, :]
    # v = singleScoreSP[0, :].unsqueeze(0)

    for i in range(1, T):
        subScore = scoreFB[i, :i, :]



        v[i] = torch.logaddexp(
                v[i-1,:] + noiseScoreFB[i-1,:],              # skip
                torch.logsumexp(v[:i, :]+subScore, dim = 0) )
        v[i] += singleScoreSP[i,:]
        # v[i] = curV
    v,q = torch.chunk(v, 2, dim = -1)

    q = torch.flip(q, (0,))


    logZ = v[-1]


    # print(logZ)
    scoreFB = None
    noiseScoreFB = None

    grad = (v.unsqueeze(0)+ ((q.unsqueeze(1)-logZ) + score))#.exp()

    # minus 2* F.softplus(diag of score)
    grad = grad - 2*torch.diag_embed(F.softplus(torch.diagonal(score, dim1 = 0, dim2 = 1)), dim1= 0, dim2 = 1)

    # grad = grad#*torch.ones(T, T, device= grad.device).tril().unsqueeze(-1)

    # do some clamping for satbility 
    # grad = 
    # grad = grad - grad*(grad>0)

     
    grad = grad*torch.ones(T, T, device= grad.device).tril().unsqueeze(-1)

    grad = grad.exp()

    grad = grad*torch.ones(T, T, device= grad.device).tril().unsqueeze(-1)

    # for grad of noise, it is similar

    
    gradNoise = (v[:-1]+ q[1:] + noiseScore-logZ)
    # gradNoise = gradNoise - gradNoise*(gradNoise>0)
    gradNoise = gradNoise.exp()

    
    # print(grad)
    # print(grad[:,:,0])
    # print("grad:", grad[:,:,0])
    # print("gradNoise:",gradNoise[:,0])


    return logZ, grad, gradNoise


class ComputeLogZFasterGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, score, noiseScore):
        logz, grad, gradNoise = forward_backward(score, noiseScore)
        ctx.save_for_backward(grad, gradNoise)

        return logz

    @staticmethod
    def backward(ctx, grad_output):
        grad, gradNoise, = ctx.saved_tensors
        assert(grad_output.shape[-1] == grad.shape[-1])
        return grad*grad_output, gradNoise*grad_output
        

computeLogZFasterGrad = ComputeLogZFasterGrad.apply


@torch.jit.script
def evalPathSlow(intervals: List[List[Tuple[int, int]]], score, noiseScore):
    # the naive version
    # print(intervals)
    result = []




    noiseScore = F.pad(noiseScore, (0,0,1,0))
    noiseScoreCum = torch.cumsum(noiseScore, dim = 0)

    for idx, curList in enumerate(intervals):
        v = noiseScoreCum[-1, idx]

        for i, j in curList:
            v = v+ (score[j,i, idx]) - noiseScoreCum[j, idx] + noiseScoreCum[i, idx]
        result.append(v)

    result = torch.stack(result, dim = -1)




    return result
    


    

def evalPath(intervals: List[List[Tuple[int, int]]], score, noiseScore):
    # score: [endpos,  beginPos, nBatch], note that the interval is close
    assert(len(score.shape) == 3)
    assert(score.shape[0] == score.shape[1])
    T = score.shape[0]
    nBatch = score.shape[2]

    # first gather then scatter_add
    # idxAll = []
    # for idx, curList in enumerate(intervals):
        # idxAll.append(idx)
        # (score[j,i, idx])

        # result.append(v)
    device = score.device

    noiseScore = F.pad(noiseScore, (0,0,1,0))
    noiseScoreCum = torch.cumsum(noiseScore, dim = 0)

    indices = [ idx+ i*nBatch + j*nBatch*T for idx, curList in enumerate(intervals) for i,j in curList]  
    batchIndices = [ idx for idx, curList in enumerate(intervals) for _ in curList]  



    indices_neg = [idx+ i*nBatch  for idx, curList in enumerate(intervals) for i,j in curList]  
    indices_pos= [ idx+ j*nBatch for idx, curList in enumerate(intervals) for i,j in curList]  


    indices = torch.tensor(indices, device = device, dtype = torch.long)
    batchIndices= torch.tensor(batchIndices, device = device, dtype = torch.long)
    indices_pos = torch.tensor(indices_pos, device = device, dtype = torch.long)
    indices_neg = torch.tensor(indices_neg, device = device, dtype = torch.long)
    gathered_pos = noiseScoreCum.view(-1).gather(0, indices_pos)
    gathered_neg= noiseScoreCum.view(-1).gather(0, indices_neg)

    gathered = score.view(-1).gather(0, indices) - (gathered_pos-gathered_neg)

    result = gathered.new_zeros(nBatch, device = device)

    result = result.scatter_add(-1, batchIndices, gathered)
    result = result+ noiseScoreCum[-1,:]

    return result


class NeuralSemiCRFInterval:
    def __init__(self, score, noiseScore):
        """ The output layer for multiple tracks of non-overlapping intervals

        arguments:
        score -- the score matrix for all possible [begin, end] pairs, which has a shape of [T, T, nBatch],
                 where T is the length of the sequence and nBatch is how many event tracks to be decoded simultaneously.
        noiseScore -- The non-event score for the internval [t, t+1], which has a shape of [T-1, nBatch]
        """
        
        self.score = score
        self.noiseScore = noiseScore
    

    def decode(self, forcedStartPos=None, forward=False):
        if forward:
            return viterbi(self.score, self.noiseScore, forcedStartPos)
        else:
            return viterbiBackward(self.score, self.noiseScore, forcedStartPos)


    def evalPath(self, intervals):
        """ compute the unnormalized score """
        pathScore = evalPath(intervals, self.score, self.noiseScore)
        return pathScore
            

    def computeLogZ(self, noBackward =False):
        """ compute the log normalization factor """
        if noBackward:
            return computeLogZ(self.score, self.noiseScore)
        else:
            return computeLogZFasterGrad(self.score, self.noiseScore)

    def logProb(self, intervals, noBackward=False):
        return self.evalPath(intervals) - self.computeLogZ(noBackward=noBackward)


if __name__ == "__main__":
    # a simple example
    
    score = ((torch.randn(200,  200, 4))).cuda()
    noiseScore= ((torch.randn(199,  4))).cuda()

    noiseScore.requires_grad_()
    score.requires_grad_()
    optimizer = torch.optim.Adam([score, noiseScore], 1e-3)
    intervals = [
            [(0,2), (4,6),(6,6), (7,8)],
            [(1,2), (3,5), (19,19)],
            [(0,0),(4,7)],
            [],
            ]
    

    for i in range(10000):
        optimizer.zero_grad()
        crf = NeuralSemiCRFInterval(score, noiseScore)
        logP  = crf.evalPath(intervals) - crf.computeLogZ()
        loss = -logP.sum()
        (loss).backward()
        optimizer.step()

        print(loss)

        if i%100==0:
            with torch.no_grad():
                recons = crf.decode()
                print(recons)

