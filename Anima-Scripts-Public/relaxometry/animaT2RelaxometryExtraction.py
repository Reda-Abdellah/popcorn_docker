#!/usr/bin/python3
# Warning: works only on unix-like systems, not win where "python animaRelaxometryGMMExtraction.py ..." has to be run

import sys
import argparse
import tempfile
import os
import shutil
from subprocess import call

if sys.version_info[0] > 2:
    import configparser as ConfParser
else:
    import ConfigParser as ConfParser

configFilePath = os.path.join(os.path.expanduser("~"), ".anima",  "config.txt")
if not os.path.exists(configFilePath):
    print('Please create a configuration file for Anima python scripts. Refer to the README')
    quit()

configParser = ConfParser.RawConfigParser()
configParser.read(configFilePath)

animaDir = configParser.get("anima-scripts", 'anima')
animaDataDir = configParser.get("anima-scripts", 'extra-data-root')
animaScriptsDir = configParser.get("anima-scripts", 'anima-scripts-public-root')

# Argument parsing
parser = argparse.ArgumentParser(
    description="From a 4D input image, computes its brain mask and evaluates the desired relaxometry maps (mono T2, "
                "multi-T2 and/or MWF) plus the B1 map.")
parser.add_argument('-i', '--input', type=str, required=True, help='4D relaxometry image file to process')
parser.add_argument('-t', '--T1', type=str, default="", help="Optional T1 map")
parser.add_argument('-m', '--image-for-mask', type=str, default="", help="High quality image for computing brain mask (if MP2RAGE, INV2 image is better)")
parser.add_argument('-e', '--echo-spacing', type=float, required=True, help="Echo spacing of the CPMG sequence")
parser.add_argument('--tr-value', type=float, default=5000, help="TR value of the acquisition (used for mono T2 initialization")
parser.add_argument('-o', '--mono-out', type=str, default="", help="Mono T2 estimation output")
parser.add_argument('-g', '--gmm-out', type=str, default="", help="Multi T2 weights estimation output")
parser.add_argument('--no-brain-masking', action='store_true', help="Do not perform any brain masking, may be much longer")

args = parser.parse_args()
if args.mono_out == "" and args.gmm_out == "":
    print('No output was specified, please specify at least one of mono-out, gmm-out')
    quit()

tmpFolder = tempfile.mkdtemp()

inputImage = args.input
inputImagePrefix = os.path.splitext(inputImage)[0]
if os.path.splitext(inputImage)[1] == '.gz':
    inputImagePrefix = os.path.splitext(inputImagePrefix)[0]

tmpInputImagePrefix = os.path.join(tmpFolder, os.path.basename(inputImagePrefix))

# Extract brain from T1 image if present (used for further processing)
maskImage = ""
if args.no_brain_masking is False:
    outputMask = ""
    if args.image_for_mask == "":
        imageToMask = tmpInputImagePrefix + "_extract.nrrd"
        firstVolumeExtractionCommand = [animaDir + "animaCropImage", "-i", inputImage, "-o", imageToMask, "-t", "0",
                                        "-T", "0"]
        call(firstVolumeExtractionCommand)

        brainExtractionCommand = ["python", animaScriptsDir + "brain_extraction/animaAtlasBasedBrainExtraction.py",
                                  "-i", imageToMask]
        call(brainExtractionCommand)

        outputMask = tmpInputImagePrefix + "_extract_brainMask.nrrd"
    else:
        imageToMask = tmpInputImagePrefix + "_extract.nrrd"
        brainExtractionCommand = ["python", animaScriptsDir + "brain_extraction/animaAtlasBasedBrainExtraction.py",
                                  "-i", args.image_for_mask]
        call(brainExtractionCommand)

        imagePrefix = os.path.splitext(args.image_for_mask)[0]
        if os.path.splitext(args.image_for_mask)[1] == '.gz':
            imagePrefix = os.path.splitext(imagePrefix)[0]

        outputMask = imagePrefix + "_brainMask.nrrd"

        # Now resample T1 on our reference volume
        xmlCommand = [animaDir + "animaTransformSerieXmlGenerator", "-i", os.path.join(animaDataDir, "id.txt"),
                      "-o", os.path.join(tmpFolder, "id.xml")]
        call(xmlCommand)

        resampleCommand = [animaDir + "animaApplyTransformSerie", "-i", outputMask, "-o",
                           os.path.join(tmpFolder, "generatorMask.nrrd"), "-t", os.path.join(tmpFolder, "id.xml"),
                           "-g", inputImage, "-n", "nearest"]
        call(resampleCommand)

        outputMask = os.path.join(tmpFolder, "generatorMask.nrrd")

    maskImage = outputMask

t1Image = ""
# Resample T1 image if it is there
if args.T1 != "":
    xmlCommand = [animaDir + "animaTransformSerieXmlGenerator", "-i", os.path.join(animaDataDir, "id.txt"),
                  "-o", os.path.join(tmpFolder, "id.xml")]
    call(xmlCommand)

    resampleCommand = [animaDir + "animaApplyTransformSerie", "-i", args.T1, "-o",
                       os.path.join(tmpFolder, "t1Resampled.nrrd"), "-t", os.path.join(tmpFolder, "id.xml"),
                       "-g", inputImage]
    call(resampleCommand)
    t1Image = os.path.join(tmpFolder, "t1Resampled.nrrd")

# Mono T2 estimation if required
if args.mono_out != "":
    outPrefix = os.path.splitext(args.mono_out)[0]
    if os.path.splitext(args.mono_out)[1] == '.gz':
        outPrefix = os.path.splitext(outPrefix)[0]

    monoT2Command = [animaDir + "animaT2EPGRelaxometryEstimation", "-i", inputImage, "-o", args.mono_out, "--tr", str(args.tr_value),
                     "-e", str(args.echo_spacing), "--out-b1", outPrefix + "_B1.nrrd", "-O", outPrefix + "_M0.nrrd"]

    if maskImage != "":
        monoT2Command = monoT2Command + ["-m", maskImage]

    if t1Image != "":
        monoT2Command = monoT2Command + ["--t1", t1Image]

    print("Running mono T2 estimation")
    call(monoT2Command)

# Multi T2 estimation
if args.gmm_out != "":
    multiT2Command = [animaDir + "animaGMMT2RelaxometryEstimation", "-i", inputImage, "-e", str(args.echo_spacing)]

    if args.gmm_out != "":
        outPrefix = os.path.splitext(args.gmm_out)[0]
        if os.path.splitext(args.gmm_out)[1] == '.gz':
            outPrefix = os.path.splitext(outPrefix)[0]

        multiT2Command = multiT2Command + ["--out-b1", outPrefix + "_B1.nrrd", "--out-m0", outPrefix + "_M0.nrrd",
                                           "-O", args.gmm_out,"-o", outPrefix + "_MWF.nrrd"]

    if maskImage != "":
        multiT2Command = multiT2Command + ["-m", maskImage]

    if t1Image != "":
        multiT2Command = multiT2Command + ["--t1", t1Image]

    print("Running multi T2 estimation")
    call(multiT2Command)

    if args.gmm_out != "":
        collapseCommand = [animaDir + "animaCollapseImage", "-i", args.gmm_out, "-o", args.gmm_out]
        call(collapseCommand)

shutil.rmtree(tmpFolder)
