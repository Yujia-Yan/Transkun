import math
import copy
import glob
import numpy as np
from pathlib import Path
import time
import os
import wave

import json
import pretty_midi
import pickle
import torch
import random
from collections import defaultdict, deque
import csv


# a local definition of the midi note object
class Note:
    def __init__(self, start, end, pitch, velocity, hasOnset=True, hasOffset=True):
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity
        self.hasOnset = hasOnset 
        self.hasOffset = hasOffset

    def __repr__(self):
        return str(self.__dict__)

def parseControlChangeSwitch(ccSeq, controlNumber, onThreshold = 64, endT = None):

    runningStatus = False

    seqEvent = []

    currentEvent = None
    currentStatus = False

    time = 0

    for c in ccSeq:
        if c.number == controlNumber:
            time = c.time
            if c.value>=onThreshold:
                currentStatus = True
            else:
                currentStatus = False
        
        if runningStatus != currentStatus:
            if currentStatus == True:
                #use negative number as pitch for the control change event
                # the velocity of a pedal is normalized to 0-1, where values smaller than off is cut off
                currentEvent = Note(time, None, -controlNumber, 127)
            else:
                currentEvent.end = time
                seqEvent.append(currentEvent)



        runningStatus = currentStatus


    if runningStatus and endT is not None:
        # process the case where the state is not closed off at the end
        # print("Warning: running status {} not closed at the end".format(controlNumber));
        currentEvent.end = max(endT, time)
        if currentEvent.end > currentEvent.start:
            seqEvent.append(currentEvent)

            

    return seqEvent

def parseEventAll(notesList, ccList, supportedCC = [64, 66, 67], extendSustainPedal = True, pedal_ext_offset = 0.0):
    
    # CC 64: sustain
    # CC 66: sostenuto
    # CC 67: una conda
    # normalize all velocity of notes

    notesList = [ Note(**n.__dict__) for n in notesList]
    notesList.sort(key = lambda x: (x.start, x.end,x.pitch))

    for n in notesList:
        assert n.start < n.end


    # get the ending time of the last note event for the missing off event at the boundary 
    lastT = max([n.end for n in notesList])



    
    if extendSustainPedal:
        # currently ignore cc 66
        sustainEvents = parseControlChangeSwitch(ccList, controlNumber = 64, endT = lastT)
        sustainEvents.sort(key = lambda x: (x.start, x.end,x.pitch))

        if pedal_ext_offset != 0.0:
            for n in sustainEvents:
                n.start += pedal_ext_offset
                n.end += pedal_ext_offset

        notesList = extendPedal(notesList, sustainEvents)
    
    else:
    # remove overlappings, als remove n.start>=n.end
        notesList = resolveOverlapping(notesList)
    validateNotes(notesList)


    eventSeqs = [notesList]
    # parse CC 64 for 

    for ccNum in supportedCC:
        ccSeq = parseControlChangeSwitch(ccList, controlNumber = ccNum, endT=lastT)
        eventSeqs.append(ccSeq)
    
    events = sum(eventSeqs, [])

    # sort all events by the beginning
    events.sort(key = lambda x: (x.start, x.end,x.pitch))



    return events
     
def extendPedal(note_events, pedal_events):
        note_events.sort(key = lambda x: (x.start, x.end,x.pitch))
        pedal_events.sort(key = lambda x: (x.start, x.end,x.pitch))
        ex_note_events = []

        idx = 0     

        buffer_dict = {}
        nIn = len(note_events)
        

        
        for note_event  in note_events:

            midi_note = note_event.pitch
            if midi_note in buffer_dict.keys():
                _idx = buffer_dict[midi_note]
                if ex_note_events[_idx].end > note_event.start:
                    ex_note_events[_idx].end = note_event.start


            for curPedal in pedal_events:
                if note_event.end< curPedal.end and note_event.end>curPedal.start:
                    note_event.end = curPedal.end

            
            buffer_dict[midi_note] = idx
            idx += 1
            ex_note_events.append(note_event)

        # print("haha")
        ex_note_events.sort(key = lambda x: (x.start, x.end,x.pitch))

        nOut = len(ex_note_events)
        assert(nOut == nIn)

        ex_note_events = resolveOverlapping(ex_note_events)
        validateNotes(ex_note_events)
        return ex_note_events

def resolveOverlapping(note_events):
    note_events.sort(key = lambda x: (x.start, x.end,x.pitch))

    ex_note_events = []

    idx = 0     

    buffer_dict = {}
    

    
    for note_event  in note_events:

        midi_note = note_event.pitch
        # note_event.end = max(note_event.start+1e-5, note_event.end)
        # note_event.end = max(note_event.start+1e-5, note_event.end)

        if midi_note in buffer_dict.keys():
            _idx = buffer_dict[midi_note]
            if ex_note_events[_idx].end > note_event.start:
                ex_note_events[_idx].end = note_event.start


        
        buffer_dict[midi_note] = idx
        idx += 1

        ex_note_events.append(note_event)

    ex_note_events.sort(key = lambda x: (x.start, x.end,x.pitch))

    # else:
        # print("overlappingOnsetOffset", note_event)

    # remove all notes that has start == end
    n1 = len(ex_note_events)
    error_notes = [n for n in ex_note_events if not n.start<n.end]
    ex_note_events = [n for n in ex_note_events if n.start<n.end]
    n2 = len(ex_note_events)
    if n1!=n2:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(error_notes)


    validateNotes(ex_note_events)
    return ex_note_events


def validateNotes(notes):
    pitches = defaultdict(list)
    for n in notes:
        if len(pitches[n.pitch])>0:
            nPrev = pitches[n.pitch][-1]
            assert n.start >= nPrev.end, str(n)+ str(nPrev)
        
        assert n.start < n.end, n

        pitches[n.pitch].append(n)



def createIndexEvents(eventList):
    from ncls import FNCLS
    # internally uses ncls package
    starts = np.array([_.start for _ in eventList])
    ends = np.array([_.end for _ in eventList])

    index = FNCLS(starts, ends, np.arange(len(eventList)))

    return index


def querySingleInterval(start, end, index):
    starts = np.array([start], dtype = np.double)
    ends = np.array([end], dtype = np.double)
    queryIds = np.array([0])
    r_id, r_loc = index.all_overlaps_both(starts, ends, queryIds)

    return r_loc


def createDataset(datasetPath, extendPedal = True):

    # create dataset 


    filenameAll = []

    samplesAll = []

    for path in Path(datasetPath).rglob('*/*.midi'):
        t1 = time.time()
        # also find the correspoding audio file

        
        midiFile = pretty_midi.PrettyMIDI(str(path))
        assert(len(midiFile.instruments) == 1)

        inst = midiFile.instruments[0]
        events = parseEventAll(inst.notes, inst.control_changes, extendPedal)

        t2 = time.time()
        # also read meta info from the wav file

        relPath = path.relative_to(datasetPath)

        # wavPath = stem.joinpath("wav")
        wavPath = path.with_suffix(".wav")

        with wave.open(str(wavPath)) as f:
            fs = f.getframerate()
            nSamples = f.getnframes()
            nChannel = f.getnchannels()

        

        # read meta information about the wav file
        print(relPath)
        print("nSamples:{} fs:{} nChannel:{}".format(nSamples, fs, nChannel))
        # print(events)

        sample = {"relPath": relPath, "nSamples":nSamples, "fs": fs, "nChannel": nChannel, "notes":  events}
        samplesAll.append(sample)

    return samplesAll

def parseMIDIFile(midiPath, extendSustainPedal=False, pedal_ext_offset = 0.0):
    # hack for the maps dataset
    pretty_midi.pretty_midi.MAX_TICK = 1e10
    midiFile = pretty_midi.PrettyMIDI(midiPath)
    assert(len(midiFile.instruments) == 1)

    inst = midiFile.instruments[0]
    events = parseEventAll(inst.notes, inst.control_changes, extendSustainPedal=extendSustainPedal, pedal_ext_offset = pedal_ext_offset)
    return events


def createDatasetMaestro(datasetPath, datasetMetaJsonPath, extendSustainPedal=True):
    datasetMetaJsonPath = Path(datasetMetaJsonPath)

    samplesAll = []

    with datasetMetaJsonPath.open() as f:
        metaInfo = json.load(f)
        for e in metaInfo:
            print(e)
            e = dict(e)
            midiPath = os.path.join(datasetPath, e["midi_filename"])
            audioPath = os.path.join(datasetPath, e["audio_filename"])
            
            midiFile = pretty_midi.PrettyMIDI(midiPath)
            assert(len(midiFile.instruments) == 1)

            inst = midiFile.instruments[0]
            if len(midiFile.instruments)>1:
                raise Exception("contains more than one track")
            events = parseEventAll(inst.notes, inst.control_changes, extendSustainPedal=extendSustainPedal)


            with wave.open(audioPath) as f:
                fs = f.getframerate()
                nSamples = f.getnframes()
                nChannel = f.getnchannels()


            e["notes"] = events
            e["fs"] = fs
            e["nSamples"] = nSamples
            e["nChannel"] = nChannel
            samplesAll.append(e)

    return samplesAll

def createDatasetMaestroCSV(datasetPath, datasetMetaCSVPath, extendSustainPedal=True):
    datasetMetaCSVPath = Path(datasetMetaCSVPath)

    samplesAll = []

    with datasetMetaCSVPath.open() as f:
        # metaInfo = json.load(f)
        metaInfo = csv.DictReader(f)
        for e in metaInfo:
            print(e)
            midiPath = os.path.join(datasetPath, e["midi_filename"])
            audioPath = os.path.join(datasetPath, e["audio_filename"])
            
            midiFile = pretty_midi.PrettyMIDI(midiPath)
            assert(len(midiFile.instruments) == 1)

            inst = midiFile.instruments[0]
            if len(midiFile.instruments)>1:
                raise Exception("contains more than one track")
            events = parseEventAll(inst.notes, inst.control_changes, extendSustainPedal=extendSustainPedal)


            with wave.open(audioPath) as f:
                fs = f.getframerate()
                nSamples = f.getnframes()
                nChannel = f.getnchannels()


            e["notes"] = events
            e["fs"] = fs
            e["nSamples"] = nSamples
            e["nChannel"] = nChannel
            samplesAll.append(e)

    return samplesAll


def readAudioSlice(audioPath, begin, end, normalize=True):
    from scipy.io import wavfile
    import scipy.io
    fs, data = wavfile.read(audioPath, mmap = True)
        
    b = math.floor((begin)*fs)
    e = math.floor(end*fs)
    dur = e - b 
    # dur = math.ceil((end-begin)*fs)
    e = b+ dur
    # print(dur)

    # handle the case where b is negative
    
    l = data.shape[0]

    if len(data.shape) == 1:
        data = data[:, np.newaxis]

    result = (data[max(b,0): min(e,l), :])

    # print("-----------")
    # print(dur, l, b, e)
    # print(result.shape)

    # handle padding
    lPad = max(-b, 0)
    rPad = max(e-l, 0)

    # print(lPad,rPad)



    # print(e-b)
    # print(result.shape)

    # normalize the audio to [-1,1] accoriding to the type
    if normalize:
        tMax = (np.iinfo(result.dtype)).max
        result = np.divide(result, tMax, dtype=np.float32)

    if lPad >0 or rPad>0:
        result = np.pad(result,  ((lPad, rPad),(0,0)), 'constant')

    return result, fs

# resolution can be as high as 32767, however it may not be supported by certain DAW
def writeMidi(notes, resolution = 960): 
    validateNotes(notes)
    # outputMidi = pretty_midi.PrettyMIDI(resolution=32767)
    outputMidi = pretty_midi.PrettyMIDI(resolution=resolution)

    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano= pretty_midi.Instrument(program = piano_program)


    for note in notes:
        if note.pitch>0:
            note = pretty_midi.Note(start = note.start,
                                    end = note.end,
                                    pitch = note.pitch,
                                    velocity = note.velocity)
            piano.notes.append(note)
        else:
            cc_on = pretty_midi.ControlChange(-note.pitch, note.velocity, note.start)
            cc_off = pretty_midi.ControlChange(-note.pitch, 0, note.end)

            piano.control_changes.append(cc_on)
            piano.control_changes.append(cc_off)
            
            


    outputMidi.instruments.append(piano)
    return outputMidi


class DatasetMaestro:
    def __init__(self, datasetPath,  datasetAnnotationPicklePath):
        self.datasetPath = datasetPath
        self.datasetAnnotationPicklePath = datasetAnnotationPicklePath

        t1 = time.time()
        print("loading the annotation file...")
        with open(datasetAnnotationPicklePath, "rb") as f:
            self.data = pickle.load(f)
        t2 = time.time()
        
        self.durations = [float(_["duration"]) for _ in self.data]
        totalTime = sum(self.durations)

        print("n:", len(self.data), " totalDuration: ", totalTime,  " elapsed:", t2-t1)


        print("creating index for notes in all pieces...")


        t1 = time.time()
        for e in self.data:
            e["index"] = createIndexEvents(e["notes"])
        
        t2 = time.time()
        print("elapsed:", t2-t1)

    def __getstate__(self):
        return {"datasetPath":self.datasetPath,  "datasetAnnotationPicklePath": self.datasetAnnotationPicklePath}

    def __setstate__(self, d):
        datasetPath = d["datasetPath"]
        datasetAnnotationPicklePath = d["datasetAnnotationPicklePath"]
        self.__init__(datasetPath, datasetAnnotationPicklePath)

    def getSample(self, idx, normalize=True):
        # for evaluation
        e=self.data[idx]

        notes = e["notes"]
        audioName = e["audio_filename"]

        audioPath = e["audio_filename"]
        audioPath = os.path.join(self.datasetPath, audioPath)
        from scipy.io import wavfile
        fs, result= wavfile.read(audioPath, mmap = False)

        if normalize:
            tMax = (np.iinfo(result.dtype)).max
            result = np.divide(result, tMax, dtype=np.float32)


        #  readAudio

        return audioName, notes, result, fs


    def getPath(self, idx):
        # for evaluation
        e=self.data[idx]

        notes = e["notes"]
        audioName = e["audio_filename"]

        audioPath = e["audio_filename"]
        audioPath = os.path.join(self.datasetPath, audioPath)
        return audioPath
     



    def fetchData(self, idx, begin, end, audioNormalize, notesStrictlyContained):
        e= self.data[idx]
        
        # fetch the notes in this interval
        if end <0 and begin<0:
            noteIndices = []
        else:
            noteIndices = querySingleInterval(max(begin,0.0), max(end,0.0), e["index"])

        notes = [e["notes"][int(_)] for _ in noteIndices]
        # print("be",begin,end)

        # for handling notes that goes beyond the current window

        if notesStrictlyContained:
            # notes = [_ for _ in notes if _.start>= begin and _.end<end]
            notes = [Note( _.start-begin,
                           _.end-begin,
                           _.pitch, _.velocity)
                           for _ in notes if _.start>=begin and _.end<end]
            
        else:
            # trim the notes by the boudnary, notes overlapping between segments will be merged during inference
            notes = [Note(max(_.start,begin) - begin,
                          min(_.end ,end) - begin, 
                          _.pitch, 
                          _.velocity,
                          _.start>=begin,
                          _.end<end)  for _ in notes]


        # for n in notes:
            # assert(n.start>=0), n
            # assert(n.end<=end-begin), n

        # fetch the corresponding audio chunk from the file

        audioPath = e["audio_filename"]
        audioPath = os.path.join(self.datasetPath, audioPath)

        audioSlice, fs = readAudioSlice(audioPath, begin,end, audioNormalize)

        # if self.transforms is not None:
            # for t in self.transforms:
                # notes, audioSlice = t(notes, audioSlice)

        return notes, audioSlice, fs

        
    def sampleSlice(self, durationInSecond, audioNormalize= True, notesStrictlyContained=True):
        # the last argument indicate whether or not to include notes that is not entirely inside the region


        # sample an entry  from the dataset 
        idx = random.choices(list(range(len(self.durations))), self.durations)[0]
        dur = self.durations[idx]

        # then sample a specific chunk inside that audio
        if dur<durationInSecond:
            begin = 0
            end = dur
        else:
            begin = random.random()*(dur-durationInSecond)
            end = begin + durationInSecond

        notes, audioSlice = self.fetchData(idx, begin,end, audioNormalize, notesStrictlyContained)

        return notes, audioSlice, fs

def sampleFromRange(valRange, log=False, triangular = False):
    l, r = valRange

    sampler = random.uniform

    if triangular:
        sampler = random.triangular

    if not log:
        v = sampler(l,r)
    else:
        l = math.log(l)
        r = math.log(r)
        v = sampler(l,r)
        v = math.exp(v)

    return v


class AugmentatorPitchShiftOnly:
    def __init__(self, sampleRate,
            pitchShiftRange = (-0.30, 0.30),
            byPassProb = 0.1
            ):
        self.sampleRate = sampleRate
        self.pitchShiftRange = pitchShiftRange
        self.byPassProb = byPassProb
    
    

    def __call__(self, x):
        if random.random()<self.byPassProb:
            return x
        nSample = x.shape[0] 
        import sox
        tfm = sox.Transformer()
        
        pitchShift = sampleFromRange(self.pitchShiftRange)
        tfm.pitch(pitchShift)

        y_out = tfm.build_array(input_array=x, sample_rate_in=self.sampleRate)

        nSampleOut  = y_out.shape[0] 

        if nSampleOut!= nSample:
            print("size changed!!")
            if nSampleOut > nSample:
                y_out = y_out[:nSample,:]
            else:
                y_out = np.pad(y_out, ((0,nSample-nSampleOut), (0,0)), 'constant')
            


        return y_out

class Augmentator:
    def __init__(self, sampleRate,
            pitchShiftRange = (-0.3, 0.3),
            reverbRange = (0,70),
            reverbRoomScale = (0, 100),
            reverbPreDelay = (0, 100),
            freqRange1 = (32, 12000),
            width_q1 = (1,4),
            gain_db1 = (-10, 5),
            noiseGain = (0, 0.01),
            contrastRange = (0, 100),
            gainRange = (0.25,4),
            # gainRange = (1, 1),
            byPassProb = 0.1,
            ):
        self.sampleRate = sampleRate
        self.pitchShiftRange = pitchShiftRange
        self.reverbRange = reverbRange
        self.reverbRoomScale = reverbRoomScale
        self.reverbPreDelay = reverbPreDelay

        # eq
        self.eqFreqRange1 = freqRange1
        self.eqWidthRange1 = width_q1
        self.eqGainRange1 = gain_db1

        self.noiseGain = noiseGain
        self.gainRange = gainRange
        self.contrastRange = contrastRange
        self.byPassProb=byPassProb
    

    def __call__(self, x):
        if random.random()<self.byPassProb:
            return x
        nSample = x.shape[0] 
        import sox
        tfm = sox.Transformer()
        
        pitchShift = sampleFromRange(self.pitchShiftRange, triangular=True)
        tfm.pitch(pitchShift)


        reverbAmount = sampleFromRange(self.reverbRange)
        roomScale = sampleFromRange(self.reverbRoomScale)
        predelay = sampleFromRange(self.reverbPreDelay)
        if reverbAmount>0:
            if random.random()>self.byPassProb:
                # print("reverb")
                tfm.reverb(reverbAmount, room_scale = roomScale, pre_delay =predelay)


        for i in range(4):
            eq1Freq = sampleFromRange(self.eqFreqRange1, log=True)
            q1 = sampleFromRange(self.eqWidthRange1)
            gain1 = sampleFromRange(self.eqGainRange1)

            if random.random()>self.byPassProb:
                # print("eq")
                tfm.equalizer(eq1Freq, q1, gain1)

        if random.random()>self.byPassProb:
            # print("compression")
            tfm.contrast(sampleFromRange(self.contrastRange))

        y_out = tfm.build_array(input_array=x, sample_rate_in=self.sampleRate)

        noiseGain = sampleFromRange(self.noiseGain)
        gain = sampleFromRange(self.gainRange, log=True)

        if random.random()<self.byPassProb:
            # print("no noise")
            noiseGain = 0


        noise =  np.random.normal(0., 1., y_out.shape).astype(np.float32)

        y_out = y_out + noiseGain*noise
        y_out = y_out*gain

        if random.random()>self.byPassProb:
            y_out = np.clip(y_out, -1,1)

        nSampleOut  = y_out.shape[0] 

        if nSampleOut!= nSample:
            print("size changed by sox!!")
            if nSampleOut > nSample:
                y_out = y_out[:nSample,:]
            else:
                y_out = np.pad(y_out, ((0,nSample-nSampleOut), (0,0)), 'constant')
            


        return y_out

class AugmentatorAudiomentations:
    def __init__(self,
                 sampleRate = 44100,
                 pitchShiftRange = (-0.2, 0.2),
                 eqDBRange =  (-3, 3),
                 snrRange = (3, 40),
                 convIRFolder = None,
                 noiseFolder = None,
                 ):
        from audiomentations import AddGaussianSNR, Compose, PitchShift,AddShortNoises, ApplyImpulseResponse, AddBackgroundNoise, SevenBandParametricEQ, PolarityInversion, Reverse

        transformList = [
                    PitchShift(*pitchShiftRange, p = 0.5),
                    SevenBandParametricEQ( *eqDBRange, p = 0.5)
                ]

        self.transform = Compose(transformList)
        
        
        if convIRFolder is not None:
            irPath = Path(convIRFolder)
            fileList = list(irPath.glob(os.path.join('**','*.wav')))
            self.reverb = ApplyImpulseResponse(fileList, p = 0.5, lru_cache_size = 2000, leave_length_unchanged = True)
            print("aug: convIR enabled")
        else:
            self.reverb = None

        transformNoiseList = []
        if noiseFolder is not None:
            noisePath = Path(noiseFolder)
            fileList = list(noisePath.glob(os.path.join('**','*.wav')))

            noiseTrans = Compose([ 
                    PolarityInversion(),
                    Reverse(),
                    ])

            transformNoiseList.append( 
                    AddBackgroundNoise(
                        fileList,
                        *snrRange,
                        p = 0.5,
                        lru_cache_size = 256,
                        noise_transform = noiseTrans))

            print("aug: noise enabled")

        transformNoiseList.append(
            AddGaussianSNR(
                    min_snr_db = snrRange[0],
                    max_snr_db = snrRange[1],
                    p = 0.5)
            )

        self.transformNoise = Compose(transformNoiseList)
        self.sampleRate = sampleRate

    def __call__(self, x):


        x = copy.deepcopy(x)
        x = x.T 

        # randomly downmix channels
        if len(x.shape) == 2:
            nChannel = x.shape[0]

            weight = 2*np.random.rand(1, nChannel)-1
            weight = (weight+1e-8)/(np.sum(np.abs(weight))+1e-8)

            x = np.matmul(weight, x)
            x = x.astype(np.float32)

        # x = x[:]
        x = x.squeeze(0)

        x = self.transform(x, sample_rate = self.sampleRate)

        # apply transform before impulse response
        if self.reverb is not None:
            xReverb = self.reverb(x, sample_rate = self.sampleRate)
            
            # randomize the wet/dry ratio
            alpha = random.random()

            x = alpha*x + (1-alpha)*xReverb



        x = self.transformNoise(x, sample_rate = self.sampleRate)

        x = x[None, :]

        x = x.T

        return x


class DatasetMaestroIterator(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 hopSizeInSecond,
                 chunkSizeInSecond,
                 audioNormalize=True,
                 notesStrictlyContained=True,
                 ditheringFrames = True,
                 seed = 1234,
                 augmentator = None):
        super().__init__()
        self.dataset = dataset
        self.hopSizeInSecond = hopSizeInSecond
        self.chunkSizeInSecond = chunkSizeInSecond
        self.audioNormalize = audioNormalize
        self.notesStrictlyContained = notesStrictlyContained
        self.ditheringFrames = ditheringFrames
        self.augmentator = augmentator


        randGen = random.Random(seed)

        chunksAll = []
        for idx, e in enumerate(self.dataset.data):
            duration = float(e["duration"])
            chunkSizeInSecond = self.chunkSizeInSecond
            hopSizeInSecond = self.hopSizeInSecond

            # split the duration into equal size chunks
            # add 1 more for safe guarding the boundary
            nChunks = math.ceil((duration+chunkSizeInSecond)/self.hopSizeInSecond)

            hopPerChunk = math.ceil(chunkSizeInSecond/self.hopSizeInSecond)

            for j in range(-hopPerChunk, nChunks+hopPerChunk):
                if self.ditheringFrames:
                    shift = randGen.random()-0.5
                else:
                    shift = 0
                begin = (j+ shift)*hopSizeInSecond - chunkSizeInSecond/2
                # end = (j+ shift)*hopSizeInSecond + chunkSizeInSecond/2
                end = begin+chunkSizeInSecond

                # if duration-begin> hopSizeInSecond:

                # add empty frames
                if begin<duration and end > 0:
                    chunksAll.append((idx, begin, end))

   
        randGen.shuffle(chunksAll)
        self.chunksAll = chunksAll                   

    def __len__(self):
        return len(self.chunksAll)

    def __getitem__(self, idx):
        if idx>self.__len__():
            raise IndexError()

        idx, begin,end = self.chunksAll[idx]
        # print(begin, end)
        notes, audioSlice, fs = self.dataset.fetchData(idx,
                               begin, 
                               end, 
                               audioNormalize = self.audioNormalize,
                               notesStrictlyContained = self.notesStrictlyContained)

        # shift all positions of notes to the begining
        # print(notes[0])
        # print(notes[-1])
        if self.augmentator is not None:
            audioSlice = self.augmentator(audioSlice)


        sample = {"notes": notes,
                  "audioSlice": audioSlice,
                  "fs": fs,
                  "begin": begin}


        return sample

def collate_fn(batch):
    return batch

def collate_fn_batching(batch):
    notesBatch = [sample["notes"] for sample in batch]
    audioSlices = [torch.from_numpy(sample["audioSlice"]) for sample in batch]

    nAudioSamplesMin = min( [_.shape[0] for _ in audioSlices])
    nAudioSamplesMax = max( [_.shape[0] for _ in audioSlices])

    assert nAudioSamplesMax-nAudioSamplesMin < 2
    
    audioSlices = [_[:nAudioSamplesMin] for _ in audioSlices]

    audioSlices = torch.stack(
            audioSlices, dim = 0)

    return {"notes": notesBatch, "audioSlices":audioSlices} 
    


def collate_fn_randmized_len(batch):
    batchNew = []
    r = random.random()*0.5 + 0.5

    for sample in batch:
        fs = sample["fs"]
        nSample = sample["audioSlice"].shape[0]
        sample["audioSlice"] = sample["audioSlice"][: math.ceil(nSample*r), :]

        T = math.ceil(nSample*r)/fs

        notes = [_ for _ in sample["notes"] if  _.end< T ]
        sample["notes"] = notes
        
        batchNew.append(sample)



    return batchNew



def midiToKeyNumber(midiNumber):
    # piano has a midi number range of [21, 108]
    # this function maps the range to [0, 87]
    return midiNumber-21

def prepareIntervalsNoQuantize(notes, targetPitch):
    validateNotes(notes)


    # tracks of intervals indexed by pitch
    # for pedal event, use a negative number
    tracks = defaultdict(list)

    # split notes into tracks and then snap to the grid
    for n in notes:
        tracks[n.pitch].append(n)


     
    # process pitch by pitch
    intervals_all = []
    velocity_all = []
    endPointRefine_all = []
    
    for p in targetPitch:
        intervals = []
        endPointRefine = []
        velocity = []
        # print("pitch:", p)
        for n in tracks[p]:
            # print(n)
            assert(n.start>=0),n.start
            assert(n.end>=0),n.end




            curVelocity = n.velocity


            tmp = ( n.start , n.end)

            intervals.append(tmp)
            endPointRefine.append((0, 0) )
            velocity.append(curVelocity)
                

        # print(intervals)
        # print(endPointRefine)
        # print(velocity)
        
        intervals_all.append(intervals)
        endPointRefine_all.append(endPointRefine)
        velocity_all.append(velocity)

        
    result = {"intervals": intervals_all, "endPointRefine": endPointRefine_all, "velocity": velocity_all}
    return result

def prepareIntervals(notes, hopSizeInSecond, targetPitch):
    validateNotes(notes)
    # print("hopSizeInSecond:", hopSizeInSecond)


    # tracks of intervals indexed by pitch
    # for pedal event, use a negative number
    tracks = defaultdict(list)

    # split notes into tracks and then snap to the grid
    for n in notes:
        tracks[n.pitch].append(n)


     
    # process pitch by pitch
    intervals_all = []
    velocity_all = []
    endPointRefine_all = []
    endPointPresence_all = []
    
    
    for p in targetPitch:
        intervals = []
        endPointRefine = []
        endPointPresence = []
        velocity = []
        # print("pitch:", p)
        for n in tracks[p]:
            # print(n)
            assert(n.start>=0),n.start
            assert(n.end>=0),n.end

            start_quantized = int(round(n.start/hopSizeInSecond))
            end_quantized = int(round(n.end/hopSizeInSecond))


            start_refine = n.start/hopSizeInSecond - start_quantized
            end_refine = n.end/hopSizeInSecond - end_quantized

            curVelocity = n.velocity


            tmp = ( start_quantized, end_quantized)
            tmpPresence = (n.hasOnset, n.hasOffset)
            # print(n)

            # check if two consecutive notes can be seaprated by interval representation
            if len(intervals)>0 and (start_quantized< intervals[-1][1] or (end_quantized == intervals[-1][1] and intervals[-1][0] == start_quantized) ):
                # raise Exception("two notes quantized in the same frame that cannot be separated: {}, {}".format(tmp, intervals[-1]))
                print("two notes quantized in the same frame that cannot be separated or they are overlapping: {}, {}. These two notes are merged".format(tmp, intervals[-1]))

                # asd
                # print(n)
                # print(intervals[-1])
                # print(start_quantized, end_quantized)

                # two consecutive note on event, treat as the same note, use the same velocity
                intervals[-1] = (intervals[-1][0], end_quantized)
                endPointRefine[-1] = (endPointRefine[-1][0], end_refine)
                endPointPresence[-1] = (endPointPresence[-1][0], n.hasOffset)
            else:
                intervals.append(tmp)
                endPointRefine.append((start_refine, end_refine) )
                endPointPresence.append(tmpPresence)
                velocity.append(curVelocity)
                

        # print(intervals)
        # print(endPointRefine)
        # print(velocity)
        
        intervals_all.append(intervals)
        endPointRefine_all.append(endPointRefine)
        endPointPresence_all.append(endPointPresence)
        velocity_all.append(velocity)

    result = {"intervals": intervals_all,
              "endPointRefine": endPointRefine_all, 
              "endPointPresence": endPointPresence_all, 
              "velocity": velocity_all}
    return result

    
