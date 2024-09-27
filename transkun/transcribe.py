import argparse

from .Data import writeMidi
import torch
import moduleconf
import numpy as np
from .Util import computeParamSize


def readAudio(path,  normalize= True):
    import pydub
    audio = pydub.AudioSegment.from_mp3(path)
    y = np.array(audio.get_array_of_samples())
    y = y.reshape(-1, audio.channels)
    if normalize:
        y =  np.float32(y)/2**15
    return audio.frame_rate, y


def main():

    import pkg_resources

    defaultWeight =  (pkg_resources.resource_filename(__name__, "pretrained/2.0.pt"))
    defaultConf =  (pkg_resources.resource_filename(__name__, "pretrained/2.0.conf"))

    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("audioPath", help = "path to the input audio file")
    argumentParser.add_argument("outPath", help = "path to the output MIDI file")
    argumentParser.add_argument("--weight", default = defaultWeight, help = "path to the pretrained weight")
    argumentParser.add_argument("--conf", default = defaultConf, help = "path to the model conf")
    argumentParser.add_argument("--device", default = "cpu", nargs= "?", help = " The device used to perform the most computations (optional), DEFAULT: cpu")
    argumentParser.add_argument("--segmentHopSize", type=float, required=False, help = " The segment hopsize for processing the entire audio file (s), DEFAULT: the value defined in model conf")
    argumentParser.add_argument("--segmentSize", type=float, required=False, help = " The segment size for processing the entire audio file (s), DEFAULT: the value defined in model conf")

    args = argumentParser.parse_args()

    path = args.weight
    device = args.device

    # TODO fix the conf
    confPath = args.conf

    confManager = moduleconf.parseFromFile(confPath)
    TransKun = confManager["Model"].module.TransKun
    conf = confManager["Model"].config


    checkpoint = torch.load(path, map_location = device)


    # conf = TransKun.Config()
    # conf.__dict__ = checkpoint['conf']

    model = TransKun(conf = conf).to(device)
    # print("#Param(M):", computeParamSize(model))


    if not "best_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["best_state_dict"], strict=False)

    model.eval()


    audioPath = args.audioPath
    outPath = args.outPath
    torch.set_grad_enabled(False)


    fs, audio= readAudio(audioPath)


    if(fs != model.fs):
        import soxr
        audio = soxr.resample(
                audio,          # 1D(mono) or 2D(frames, channels) array input
                fs,      # input samplerate
                model.fs# target samplerate
)



    x = torch.from_numpy(audio).to(device)


    notesEst = model.transcribe(x, stepInSecond=args.segmentHopSize, segmentSizeInSecond=args.segmentSize, discardSecondHalf=False)

    outputMidi = writeMidi(notesEst)
    outputMidi.write(outPath)


if __name__ == "__main__":
    main()
