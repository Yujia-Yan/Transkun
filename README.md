# Piano Transcription with Neural Semi-CRF

This repo contains code for the paper for transcribing expressive piano performance into MIDI.

Yujia Yan, Frank Cwitkowitz, Zhiyao Duan, Skipping the Frame-Level: Event-Based Piano Transcription With Neural Semi-CRFs, Advances in Neural Information Processing Systems, 2021, 

[OpenReview](https://openreview.net/forum?id=DGA8XbJ8FVd), [paper](https://openreview.net/pdf?id=DGA8XbJ8FVd), [appendix](https://openreview.net/attachment?id=DGA8XbJ8FVd&name=supplementary_material)


## pip installation

```bash
pip3 install transkun
```

The pip package provides a quick command for transcribing piano performance audio into midi:

```bash
$ transkun input.mp3 output.mid
```

with cuda:

```bash
$ transkun input.mp3 output.mid --device cuda
```


## Overview
<img width="1405" alt="image" src="https://user-images.githubusercontent.com/1996534/183318064-db32dbef-500d-4710-93a1-10acd3eb8825.png">

This system works as follows: 
1. A score tensor is computed from the input audio that scores every possible intervals for whether or not being an event.
2. The score tensor computed in (1) is then decoded by the proposed semi-CRF layer to obtain event intervals directly via dynamic programming (viterbi).
3. Attributes, e.g., velocity and refined onset/offset position, associated with each interval are then predicted from the extracted event intervals from (2).

## Basic Usage

### The Semi-CRF Layer

This code includes an neural semi-CRF module that is optimized for the problem domain.  

Here is a minimal example for using this module:

```python
import CRF
import torch

T = 200
NBatch = 4

# representing the score for the interval [TBegin, TEnd]
# dimensions: [TEnd, TBegin, NBatch]
# only the lower triangular part is used
score = ((torch.randn(T,  T, NBatch))).cuda()

# representing the score for being not an interval, dimensions [TBegin, TBegin+1]
noiseScore= ((torch.randn(T-1,  NBatch))).cuda()

# a list of list of non-overlapping intervals
intervals = [
        [(0,2), (4,6),(6,6), (7,8)],
        [(1,2), (3,5), (19,19)],
        [(0,0),(4,7)],
        [],
        ]

crf = CRF.NeuralSemiCRFInterval(score, noiseScore)

## log probability
logP = crf.logProb(intervals)

## decoding
decoded = crf.decode()

## decoding starting from a given position, useful for segment based processing
decoded = crf.decode(forcedStartPos = [4]*NBatch)
```

### Transcribing piano performance into a MIDI file

```bash
python3 -m transkun.transcribe -h

usage: transkun [-h] [--weight WEIGHT] [--device [DEVICE]] [--segmentHopSize SEGMENTHOPSIZE] [--segmentSize SEGMENTSIZE] audioPath outPath

positional arguments:
  audioPath             path to the input audio file
  outPath               path to the output MIDI file

optional arguments:
  -h, --help            show this help message and exit
  --weight WEIGHT       path to the pretrained weight
  --device [DEVICE]     The device used to perform the most computations (optional), DEFAULT: cpu
  --segmentHopSize SEGMENTHOPSIZE
                        The segment hopsize for processing the entire audio file (s), DEFAULT: 10
  --segmentSize SEGMENTSIZE
                        The segment size for processing the entire audio file (s), DEFAULT: 20
```

Please note that segmentHopSize and segmentSize should be chosen such that there is an overlap between consecutive segments. 
The included weight is trained under segmentHopSize = 10 and segmentSize = 20.

This script can also be used directly as the command line command 'transkun' if the pip package is installed, e.g., 

```bash
$ transkun input.mp3 output.mid
```


## Handling the dataset

### Converting to the same sampling rate

We assume the data contains only the same sampling rate 44100hz. Therefore for the maestro dataset it is necessary to perform sampling rate conversion to 44100hz for the last two years (2017 and 2018) .  

#### Justification for using the original sample rate

Many exisiting works downsample the original audio files to a smaller sampling rate. And it seems quite "conventional" to do downsampling. However, from our perspective, this step is unnecessary:

1. Spectral processing for getting the mel spectrum only incurs a small computation cost. However, a good quality downsampling algorithm is more expensive than computing the mel spectrum.

2. Whether or not compute the spectrum from the downsampled version, the size of the input to the neural network is at the same range. 

3. There is no need to load all data into memory as the wav file is random accessible. This work directly random acesses the wav file instead of using a specific file format. 

### Generating metadata files

Assuming all audio files have already been converted to the same sampling rate, we iterate the entire dataset to combine the groundtruth midi and metadata into a single file.

The following script will generate train.pt, val.pt and test.pt

```bash
python3 -m transkun.createDatasetMaestro -h
usage: createDatasetMaestro.py [-h] [--noPedalExtension] datasetPath metadataCSVPath outputPath

positional arguments:
  datasetPath         folder path to the maestro dataset
  metadataCSVPath     path to the metadata file of the maestro dataset (csv)
  outputPath          path to the output folder

optional arguments:
  -h, --help          show this help message and exit
  --noPedalExtension  Do not perform pedal extension according to the sustain pedal
```

This command will generate train.pt, dev.pt, test.pt in the outputPath.

## Training

After generating the medata files, we can perform training using the dataset. During training, the audio waveforms will be fetched directly from the original .wav files.

Firstly, generate a config template file for the model:

```bash
mkdir checkpoint
python3 -m transkun.generateConfig transkun.Model_ablation > checkpoint/conf.json
```

Then we call the training script.

```bash
python3 -m transkun.train -h
```

## The Evaluation Module

### Comparing the output MIDI files and the groundtruth MIDI files

We also provide an out-of-box tool for computing metrics directly from output midi files.

```bash
usage: computeMetrics.py [-h] [--outputJSON OUTPUTJSON] [--noPedalExtension] [--nProcess [NPROCESS]] [--computeDeviations] estDIR groundTruthDIR

compute metrics directly from MIDI files.
Note that estDIR should have the same folder structure as the groundTruthDIR.
The MIDI files to evaluate should have the same extension as the ground truth.
Metrics outputed are ordered by precision, recall, f1, overlap.

positional arguments:
  estDIR
  groundTruthDIR

optional arguments:
  -h, --help            show this help message and exit
  --outputJSON OUTPUTJSON
                        path to save the output file for detailed metrics per audio file
  --noPedalExtension    Do not perform pedal extension according to the sustain pedal for the ground truch
  --nProcess [NPROCESS]
                        number of workers for multiprocessing
  --computeDeviations   output detailed onset/offset deviations for each matched note.
```

Currently, we do not support evaluation of multitrack MIDIs.

This command can also be used directly as the command line script 'transkunEval' if the pip package is installed.

### Ploting the empirical cumulative distribution function to visualize the onset/offset accuracy

Use the following script to plot the ECDF curve for onset/offset deviations:

```bash
usage: plotDeviation.py [-h] [--labels [LABELS [LABELS ...]]] [--offset] [--T T] [--output [OUTPUT]] [--noDisplay] evalJsons [evalJsons ...]

plot the empirical cumulative distribution function on onset/offset deviations

positional arguments:
  evalJsons             a seqeunce of the output json files from the computeMetrics script, the deviation output should be enabled

optional arguments:
  -h, --help            show this help message and exit
  --labels [LABELS [LABELS ...]]
                        specify labels to show on the legend
  --offset              plot the offset deviation curve. If not specified, onset deviation curve will be plotted
  --T T                 time limit(ms), default: 50ms
  --output [OUTPUT]     filename to save
  --noDisplay           Do not show the figure.
```

![loading-ag-1328](./assets/exampleDev.png)

### Citation

If you find this repository helpful, please consider citing:


Bibtex:

```bibtex
@inproceedings{
yan2021skipping,
title={Skipping the Frame-Level: Event-Based Piano Transcription With Neural Semi-{CRF}s},
author={Yujia Yan and Frank Cwitkowitz and Zhiyao Duan},
booktitle={Advances in Neural Information Processing Systems},
editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
year={2021},
url={https://openreview.net/forum?id=AzB2Pq7UFsA}
}
```
