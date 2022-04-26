from . import Data
import pickle
import argparse
import pickle
import os

if __name__ == "__main__":
    
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("datasetPath", help = "folder path to the maestro dataset")
    argumentParser.add_argument("metadataCSVPath", help = "path to the metadata file of the maestro dataset (csv)")
    argumentParser.add_argument("outputPath", help = "path to the output folder")
    argumentParser.add_argument("--noPedalExtension", action='store_true', help = "Do not perform pedal extension according to the sustain pedal")


    args = argumentParser.parse_args()

    datasetPath = args.datasetPath
    datasetMetaCSVPath = args.metadataCSVPath
    extendPedal = not args.noPedalExtension
    outputPath = args.outputPath




    dataset = Data.createDatasetMaestroCSV(datasetPath, datasetMetaCSVPath, extendSustainPedal = extendPedal)

    train = []
    val= []
    test = []
    for e in dataset:
        if e["split"] == "train":
            train.append(e)
        elif e["split"] == "validation":
            val.append(e)
        elif e["split"] == "test":
            test.append(e)

    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    with open(os.path.join(outputPath, 'train.pickle'), 'wb') as f:
        pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)


    with open(os.path.join(outputPath, 'val.pickle'), 'wb') as f:
        pickle.dump(val, f, pickle.HIGHEST_PROTOCOL)


    with open(os.path.join(outputPath, 'test.pickle'), 'wb') as f:
        pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)
