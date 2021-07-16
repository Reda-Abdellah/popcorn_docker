#!/usr/bin/python3
# Warning: works only on unix-like systems, not windows where "python animaMultiCompartmentModelEstimation.py ..." has to be run

import sys
import argparse

if sys.version_info[0] > 2:
    import configparser as ConfParser
else:
    import ConfigParser as ConfParser

import os
from subprocess import call

configFilePath = os.path.join(os.path.expanduser("~"), ".anima",  "config.txt")
if not os.path.exists(configFilePath):
    print('Please create a configuration file for Anima python scripts. Refer to the README')
    quit()

configParser = ConfParser.RawConfigParser()
configParser.read(configFilePath)

animaDir = configParser.get("anima-scripts", 'anima')

# Argument parsing
parser = argparse.ArgumentParser(
    description="Performs multi-compartment models estimation and averaging from pre-processed or unprocessed DWI image(s)")
parser.add_argument('-t', '--type', type=str, default="tensor",
                    help="Type of compartment model for fascicles (stick, zeppelin, tensor, noddi, ddi)")
parser.add_argument('-n', '--num-compartments', type=int, default=3,
                    help="Number of fascicle compartments at most")

parser.add_argument('--hcp', action='store_true',
                    help="Use HCP adapted models (with isotropic restricted and stationary water)")
parser.add_argument('--no-model-simplification', action='store_true',
                    help="Do not perform any model selection or averaging")
parser.add_argument('-S', '--model-selection', action='store_true',
                    help="Perform model selection instead of model averaging")

parser.add_argument('-i', '--input', type=str, required=True, help='DWI file to process')
parser.add_argument('-b', '--bval', type=str, required=True, help='DWI b-value or bval file')
parser.add_argument('-g', '--bvec', type=str, required=True, help='DWI gradients file')
parser.add_argument('-m', '--mask', type=str, default="", help='Computation mask')

args = parser.parse_args()

# Get parameters from arguments parser
baseEstimationCommand = [animaDir + "animaMCMEstimator", "-FR"]
if args.type.lower() == "ddi":
    baseEstimationCommand += ["--optimizer", "bobyqa", "--ml-mode", "1"]
else:
    baseEstimationCommand += ["--optimizer", "levenberg", "--ml-mode", "2"]

if args.hcp is True:
    baseEstimationCommand += ["-S"]

if args.no_model_simplification is False:
    if args.model_selection is True:
        baseEstimationCommand += ["-M"]

# Default model is stick
modelNumber = 1
if args.type.lower() == "zeppelin":
    modelNumber = 2
elif args.type.lower() == "tensor":
    modelNumber = 3
elif args.type.lower() == "noddi":
    modelNumber = 4
elif args.type.lower() == "ddi":
    modelNumber = 5

baseEstimationCommand += ["-c", str(modelNumber)]

dwiImage = args.input
print("Computing MCM for image " + dwiImage)

dwiImagePrefix = os.path.splitext(dwiImage)[0]
if os.path.splitext(dwiImage)[1] == '.gz':
    dwiImagePrefix = os.path.splitext(dwiImagePrefix)[0]

estimationCommandWithInputs = baseEstimationCommand + ["-i", dwiImage, "-b", args.bval, "-g", args.bvec]
if not (args.mask == ""):
    estimationCommandWithInputs += ["-m", args.mask]

if (args.no_model_simplification is False) and (args.model_selection is False):
    # Perform all estimations and model averaging
    mergeDataFile = open(dwiImagePrefix + "_MCM_List.txt", 'w')
    mergeDataAICFile = open(dwiImagePrefix + "_MCM_AIC_List.txt", 'w')
    mergeDataB0File = open(dwiImagePrefix + "_MCM_B0_List.txt", 'w')
    mergeDataS2File = open(dwiImagePrefix + "_MCM_S2_List.txt", 'w')

    for numCompartments in range(0, args.num_compartments + 1):
        outputPrefix = dwiImagePrefix + "_MCM_N" + str(numCompartments)
        estimationCommand = estimationCommandWithInputs + ["-o", outputPrefix + ".mcm", "-a",
                                                           outputPrefix + "_aic.nrrd", "--out-b0",
                                                           outputPrefix + "_B0.nrrd", "--out-sig",
                                                           outputPrefix + "_S2.nrrd", "-n", str(numCompartments)]
        call(estimationCommand)

        mergeDataFile.write(outputPrefix + ".mcm\n")
        mergeDataAICFile.write(outputPrefix + "_aic.nrrd\n")
        mergeDataB0File.write(outputPrefix + "_B0.nrrd\n")
        mergeDataS2File.write(outputPrefix + "_S2.nrrd\n")

    mergeDataFile.close()
    mergeDataAICFile.close()
    mergeDataB0File.close()
    mergeDataS2File.close()

    averagingCommand = [animaDir + "animaMCMModelAveraging", "-i", dwiImagePrefix + "_MCM_List.txt", "-b",
                        dwiImagePrefix + "_MCM_B0_List.txt", "-n", dwiImagePrefix + "_MCM_S2_List.txt", "-a",
                        dwiImagePrefix + "_MCM_AIC_List.txt", "-o",
                        dwiImagePrefix + "_MCM_avg.mcm", "-O", dwiImagePrefix + "_MCM_B0_avg.nrrd", "-N",
                        dwiImagePrefix + "_MCM_S2_avg.nrrd", "-m",
                        dwiImagePrefix + "_MCM_mose_avg.nrrd", "-C"]
    call(averagingCommand)

else:
    if args.no_model_simplification is True:
        outputPrefix = dwiImagePrefix + "_MCM_N" + str(args.num_compartments)
    else:
        outputPrefix = dwiImagePrefix + "_MCM_MS" + str(args.num_compartments)

    estimationCommand = estimationCommandWithInputs + ["-o", outputPrefix + ".mcm", "-a", outputPrefix + "_aic.nrrd",
                                                       "--out-b0", outputPrefix + "_B0.nrrd", "--out-sig",
                                                       outputPrefix + "_S2.nrrd", "-n", str(args.num_compartments)]

    if args.no_model_simplification is False:
        estimationCommand += ["--out-mose", outputPrefix + "_mose.nrrd"]

    call(estimationCommand)
