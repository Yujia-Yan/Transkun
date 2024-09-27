import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def computeParamSize(module):
    total_params = sum(p.numel() for p in module.parameters())

    # Convert to millions
    total_params_millions = total_params / 1e6
    return total_params_millions

def checkpointByPass(f, *args):
    return f(*args)

def checkpointSequentialByPass(f,n, *args):
    return f(*args)

def makeFrame(x, hopSize, windowSize, leftPaddingHalfFrame =True):
    assert(hopSize<windowSize)

    nFrame = math.ceil((x.shape[-1])/hopSize)+1

    if leftPaddingHalfFrame:
        lPad = windowSize//2
        rPad = (nFrame-1)*hopSize+windowSize//2 - x.shape[-1]
    else:
        lPad = 0
        rPad = (nFrame-1)*hopSize+windowSize - x.shape[-1]



    x = F.pad(x, (lPad, rPad))


    frames = x.unfold(-1, windowSize, hopSize)

    assert(frames.shape[-2] == nFrame), (frames.shape[-2], nFrame)

    
    return frames



class GaussianWindows(nn.Module):
    def __init__(self, n, nWin):
        super().__init__()
        self.n = n
        self.nWin=nWin

        self.sigma = nn.Parameter(
                -torch.ones(n)*1
                )

        self.center = nn.Parameter(
                torch.Tensor(
                    torch.logit((torch.arange(1, n+1))/(n+1))
                    ))
    
    def get(self):
        sigma = torch.sigmoid(self.sigma)
        center = torch.sigmoid(self.center)

        device = next(self.parameters()).device

        x= torch.arange(self.nWin, device =device)
        Y = (-0.5* ((x.unsqueeze(1)- self.nWin*center)/(sigma*self.nWin/2))**2  ).exp()

        return Y


    



class Spectrum(nn.Module):
    def __init__(self, windowSize, nExtraWins = 0,  log=False):
        super().__init__()


        self.outputDim = windowSize//2+1

        self.nChannel = (nExtraWins+1)

        self.log = log
        
        # learnable window function
        self.register_buffer( "win", torch.hann_window(windowSize))

        if nExtraWins> 0:
            self.winGen = GaussianWindows(nExtraWins, windowSize)


        self.nExtraWins = nExtraWins


    

    def forward(self, frames):

        if self.nExtraWins>0:
            wins = self.winGen.get()
            wins = torch.cat([self.win.unsqueeze(0), wins.t()], dim = 0)
        else:
            # wins = self.win.unsqueeze(0)
            wins = torch.cat([self.win.unsqueeze(0)], dim = 0)

        spectrogram = torch.fft.rfft(
                # torch.fft.fftshift(frames*self.win),
                (frames.unsqueeze(-2)*wins),
                norm= "ortho")

        # spectrogram = spectrogram.transpose(-1,-2)

        if self.log:
            spectrogram = torch.complex(spectrogram.abs(),  spectrogram.angle())

        result = spectrogram
        result = result.transpose(-1,-2) 


        return result

class MelSpectrum(nn.Module):
    def __init__(self, windowSize, f_min, f_max, n_mels, fs, nExtraWins=0, log=False, eps = 1e-5, toMono=False):
        super().__init__()


        self.outputDim = n_mels
        self.nChannel = (nExtraWins+1)

        import torchaudio
        self.register_buffer("freq2mels", torchaudio.functional.melscale_fbanks( 
                                                n_freqs = windowSize//2+1, 
                                                f_min = f_min,
                                                f_max = f_max,
                                                n_mels = n_mels,
                                                sample_rate = fs,
                                                ))
        self.log = log 
        self.eps = eps

        self.spectrogramExtractor = Spectrum(windowSize, nExtraWins)
        self.toMono = toMono


    

    def forward(self, frames):
        # output format: (.,  #frame, #freqBin, #featureChannel)

        spectrogram = self.spectrogramExtractor(frames)

        spectrogram = (spectrogram).abs().pow(2)

        if self.toMono and len(spectrogram.shape)>=4:
            spectrogram = spectrogram.mean(dim = -4, keepdim = True)


        mel = (spectrogram.transpose(-1,-2)@ self.freq2mels).transpose(-1,-2)

        if self.log:
            # with normalization
            eps = self.eps
            mel = ((mel+eps).log()-math.log(eps))/(-math.log(eps))
        

        return mel


def listToIdx(l):
    batchIndices = [ idx for idx, curList in enumerate(l) for _ in curList]  

    return batchIndices


