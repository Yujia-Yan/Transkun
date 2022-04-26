
import torch
from .Model_ablation import *


import argparse


def readAudio(path,  normalize= True):
    import pydub
    audio = pydub.AudioSegment.from_mp3(path)
    y = np.array(audio.get_array_of_samples())
    y = y.reshape(-1, audio.channels)
    if normalize:
        y =  np.float32(y)/2**15
    return audio.frame_rate, y


if __name__ == "__main__":
    argumentParser = argparse.ArgumentParser()

    argumentParser.add_argument("audioPath", help = "path to the input audio file")
    argumentParser.add_argument("outPath", help = "path to the output MIDI file")
    argumentParser.add_argument("--weight", help = "path to the pretrained weight")
    argumentParser.add_argument("--device", default = "cpu", nargs= "?", help = " The device used to perform the most computations (optional)")
    argumentParser.add_argument("--segmentHopSize", type=float, default = 10, help = " The segment hopsize for processing the entire audio file (s)")
    argumentParser.add_argument("--segmentSize", type=float, default = 20, help = " The segment size for processing the entire audio file (s)")

    args = argumentParser.parse_args()

    # path = "./ablations_scaling.pt"
    path = args.weight
    device = args.device
    checkpoint = torch.load(path, map_location = device)


    conf = TransKun.Config()
    conf.__dict__ = checkpoint['conf']

    model = TransKun(conf = conf).to(device)

    if not "best_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["best_state_dict"], strict=False)

    model.eval()


    audioPath = args.audioPath
    outPath = args.outPath
    torch.set_grad_enabled(False)


    fs, audio= readAudio(audioPath)


    assert(fs == model.fs)


    x = torch.from_numpy(audio).to(device)


    notesEst = model.transcribe(x, stepInSecond=args.segmentHopSize, segmentSizeInSecond=args.segmentSize, discardSecondHalf=True)

    outputMidi = writeMidi(notesEst)
    outputMidi.write(outPath)



