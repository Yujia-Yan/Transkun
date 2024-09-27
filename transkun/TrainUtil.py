import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_optimizer as optim
import torch.nn as nn
import numpy as np
import copy
from .LayersTransformer import LearnableSpatialPositionEmbedding



class MovingBuffer:

    def __init__(self, initValue = None, maxLen= None):
        from collections import deque
        self.values = deque(maxlen = maxLen)
        if initValue is not None:
            self.step(initValue)
        
    def step(self, value):
        self.values.append(value)

    
    def getQuantile(self, quantile):
        return float(np.quantile(self.values, q = quantile))

def checkNoneGradient(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print("Warning: detected parameter with no gradient that requires gradient:")
            print(param)
            print(param.shape)
            print(name)


def average_gradients(model, c = None, parallel = True):
    if parallel:
        size = float(dist.get_world_size())
        if c is None:
            c = size

        # size = float(dist.get_world_size())
        checkNoneGradient(model)
        for param in model.parameters():
            if param.requires_grad:
                    # print(param)
                    # print(param.shape)
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                # param.grad.data /= c 
    else:
        checkNoneGradient(model)
        if c is None:
            c = 1
        for param in model.parameters():
            if param.requires_grad:
                param.grad.data /= c 

def load_state_dict_tolerant(model, state_dict):
    model_dict =model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in state_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def save_checkpoint(filename, epoch, nIter, model,  lossTracker, best_state_dict, optimizer, lrScheduler):
    checkpoint = {
            #'conf':model.conf.__dict__,
            'state_dict': model.state_dict(), 
            'best_state_dict': best_state_dict,
            'epoch':epoch,
            'nIter': nIter,
            'loss_tracker': lossTracker,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lrScheduler.state_dict()
            }
    torch.save(checkpoint, filename)

def getOptimizerGroup(model):
    param_optimizer = list(model.named_parameters())
    # exclude GroupNorm and PositionEmbedding from weight decay
    # no_decay = ['bias', 'LayerNorm', 'GroupNorm, PositionEmbedding']  # Specify the names of parameters to exclude from weight decay
    # optimizerConfig = [
        # # Parameters with weight decay
        # {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        # # Parameters without weight decay
        # {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # print([n for n, p in param_optimizer if any(nd in n for nd in no_decay)])

    noDecay = []
    for name, module in model.named_modules():
        if isinstance(module, nn.GroupNorm) \
                or isinstance(module, nn.LayerNorm) \
                or isinstance(module, LearnableSpatialPositionEmbedding):
            noDecay.extend(list(module.parameters()))
        else:
            noDecay.extend([p for n, p in module.named_parameters() if "bias" in n])
    
    otherParams  =set(model.parameters()) - set(noDecay)
    otherParams = [param for param in model.parameters() if param in otherParams]
    noDecay = set(noDecay)
    noDecay = [param for param in model.parameters() if param in noDecay]


    optimizerConfig = [{"params": otherParams},
                        {"params": noDecay, "weight_decay":0e-7}]

    return optimizerConfig

    
def initializeCheckpoint(Model,
                         device,
                         max_lr,
                         weight_decay,
                         nIter,
                         conf):

    # conf = Model.Config()
    # if confDict is not None:
        # conf.__dict__ = confDict

    model = Model(conf).to(device)

    optimizerGroup = getOptimizerGroup(model)


    optimizer = optim.AdaBelief(
                            # model.parameters(),
                            optimizerGroup,
                            max_lr,
                            weight_decouple=True,
                            eps = 1e-8,
                            weight_decay=weight_decay,
                            rectify=True)


    lrScheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, nIter, pct_start = 0.05, cycle_momentum=False, final_div_factor = 2, div_factor = 20)

    lossTracker = {'train': [], 'val': []}
    best_state_dict = copy.deepcopy(model.state_dict())
    startEpoch = 0 
    startIter = 0

    
    return  startEpoch, startIter, model, lossTracker, best_state_dict, optimizer, lrScheduler
    

def load_checkpoint(Model, conf, filename,device, strict=False):
    checkpoint = torch.load(filename, map_location=device)

    startEpoch = checkpoint['epoch']
    startIter = checkpoint['nIter']

    # conf_dict = checkpoint['conf']
    

    # conf = Model.Config()
    # conf.__dict__ = conf_dict


    model = Model(conf = conf).to(device)

    optimizerGroup = getOptimizerGroup(model)
    
    optimizer = optim.AdaBelief(
                            # model.parameters(),
                            optimizerGroup,
                            1e-5,
                            weight_decouple=True,
                            eps = 1e-8,
                            weight_decay = 1e-2,
                            rectify=True)


    # lrScheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 2e-4, 100000, pct_start = 0.05, cycle_momentum=False, final_div_factor=2, div_factor = 20)
    # lrScheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 4e-4, 500000, pct_start = 0.05, cycle_momentum=False, final_div_factor=2, div_factor = 20)
    lrScheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 4e-4, 500000, pct_start = 0.05, cycle_momentum=False, final_div_factor=2, div_factor = 20)
     
    # debugging flag
    restartFromTheBest = False

    if restartFromTheBest:
        if not strict:
            load_state_dict_tolerant(model, checkpoint['best_state_dict'])
        else:
            model.load_state_dict(checkpoint['best_state_dict'])
        # lrScheduler.load_state_dict( checkpoint['lr_scheduler_state_dict'])
    else:
        if not strict:
            load_state_dict_tolerant(model, checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict( checkpoint['optimizer_state_dict'])

    # lrScheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 4e-4, 500000, pct_start = 0.05, cycle_momentum=False, final_div_factor=2, div_factor = 20, last_epoch = startIter)
        lrScheduler.load_state_dict( checkpoint['lr_scheduler_state_dict'])

    # lrScheduler.total_steps = 500000
    # print(optimizer.param_groups)
    best_state_dict = checkpoint['best_state_dict']

    lossTracker = checkpoint['loss_tracker']

    return startEpoch, startIter, model, lossTracker, best_state_dict, optimizer, lrScheduler


def computeMetrics(model, x, notes):
    with torch.no_grad():
        logp = model.log_prob(x, notes)
        logp= (logp.sum(-1).mean()).item()

        stats = model.computeStatsMIREVAL(x, notes)

    
    length = x.shape[1]

    nGT = stats["nGT"]
    nEst = stats["nEst"]
    nCorrect = stats["nCorrect"]
    
    result = {"logProb": logp, "length": length, "nGT": nGT, "nEst":nEst, "nCorrect":nCorrect}

    return result


def doValidation(model, dataset, parallel, device):

    resultAll = []

    with torch.no_grad():
        for idx, batch in enumerate(dataset):
            notesBatch = [sample["notes"] for sample in batch]
            audioSlices = torch.stack(
                    [torch.from_numpy(sample["audioSlice"]) for sample in batch], dim = 0) . to(device)

            result = computeMetrics(model, audioSlices, notesBatch)
            resultAll.append(result)
            print(result, "progress:{:0.2f}".format(idx/len(dataset)))


    
    # aggregate the result 
    logPAgg = sum([e["logProb"] for e in resultAll])
    lengthAgg = sum([e["length"]/model.fs for e in resultAll])
    nGT = sum([e["nGT"] for e in resultAll])
    nEst = sum([e["nEst"] for e in resultAll])
    nCorrect= sum([e["nCorrect"] for e in resultAll])

    if parallel:
        import torch.distributed as dist

        result = torch.Tensor([logPAgg, lengthAgg, nGT,nEst, nCorrect]).cuda()
        dist.all_reduce(result.data)
        logPAgg = float(result[0])
        lengthAgg = float(result[1])
        nGT = float(result[2])
        nEst= float(result[3])
        nCorrect= float(result[4])

     
    meanNLLPerSecond = -logPAgg/lengthAgg
    precision = nCorrect/nEst
    recall = nCorrect/nGT
    f1 = 2* precision*recall/(precision+recall)              


    return {"meanNLL": meanNLLPerSecond, "precision": precision, "recall":recall, "f1": f1}


