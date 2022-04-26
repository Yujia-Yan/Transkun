import json
import argparse
import importlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("moduleNames", nargs="*" )

    args = parser.parse_args()

    config = dict()


    for moduleName in args.moduleNames:
        Module = importlib.import_module(moduleName)

        moduleConfig = Module.Config()

        config[moduleName] = moduleConfig.__dict__


    configText = json.dumps(config, indent='\t')
    print(configText)


