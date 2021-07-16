#!/usr/bin/python3
# Warning: works only on unix-like systems, not windows where "python animaAnatomicalMergeImages.py ..." has to be run

import argparse
import os
import sys
import glob
from subprocess import call
import shutil

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

# Argument parsing
parser = argparse.ArgumentParser(
    description="Builds a new average anatomical form from previously registered images.")
parser.add_argument('-d', '--ref-dir', type=str, required=True, help='Reference (working) folder')
parser.add_argument('-r', '--ref-image', type=str, required=True, help='Reference image')
parser.add_argument('-e', '--files-extension', type=str, required=True, help='Input files extension')
parser.add_argument('-B', '--prefix-base', type=str, required=True, help='Prefix base')
parser.add_argument('-p', '--prefix', type=str, required=True, help='Prefix')
parser.add_argument('-w', '--weights', type=str, default="", help='Weights text file')
parser.add_argument('-b', '--bch-order', type=int, default=2, help='BCH order when composing transformations in rigid unbiased (default: 2)')
parser.add_argument('-n', '--num-images', type=int, required=True, help='Number of images')
parser.add_argument('-i', '--num-iter', type=int, required=True, help='Iteration number of atlas creation')
parser.add_argument('-c', '--num-cores', type=int, default=40, help='Number of cores to run on')

args = parser.parse_args()
os.chdir(args.ref_dir)

animaCreateImage = os.path.join(animaDir,"animaCreateImage")
animaAverageImages = os.path.join(animaDir,"animaAverageImages")
animaImageArithmetic = os.path.join(animaDir,"animaImageArithmetic")
animaTransformSerieXmlGenerator = os.path.join(animaDir,"animaTransformSerieXmlGenerator")
animaTensorApplyTransformSerie = os.path.join(animaDir,"animaTensorApplyTransformSerie")
animaComputeDTIScalarMaps = os.path.join(animaDir,"animaComputeDTIScalarMaps")
animaThrImage = os.path.join(animaDir,"animaThrImage")
animaMaskImage = os.path.join(animaDir,"animaMaskImage")

# test if all images are here
nimTest = args.num_images
if args.num_iter == 0:
    nimTest -= 1

numData = len(glob.glob(os.path.join("residualDir",args.prefix + "_*_flag")))
while numData < nimTest:
    print("Missing data " + str(numData) + " " + str(nimTest))
    call(["sleep","600"])
    numData = len(glob.glob(os.path.join("residualDir", args.prefix + "_*_flag")))

# if ok proceed
if args.num_iter == 0:
    # Write identity transform
    myfile = open(os.path.join("tempDir",args.prefix + "_1_linear_tr.txt"),"w")
    myfile.write("#Insight Transform File V1.0\n")
    myfile.write("# Transform 0\n")
    myfile.write("Transform: AffineTransform_double_3_3\n")
    myfile.write("Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\n")
    myfile.write("FixedParameters: 0 0 0\n")
    myfile.close()

    command = [animaCreateImage,"-o",os.path.join("tempDir",args.prefix + "_1_nonlinear_tr.nrrd"),
               "-b","0","-g",os.path.join(args.prefix_base,args.prefix + "_1" + args.files_extension),"-v","3"]
    call(command)

myfile = open("sumNonlinear.txt","w")
for a in range(1,args.num_images + 1):
    myfile.write(os.path.join("tempDir",args.prefix + "_" + str(a) + "_nonlinear_tr.nrrd") + "\n")

myfile.close()

command = [animaAverageImages, "-i", "sumNonlinear.txt","-o",os.path.join("residualDir","sumNonlinear_tr.nrrd")]
if not args.weights == "":
    command += ["-w",args.weights]

call(command)

command = [animaImageArithmetic,"-i",os.path.join("residualDir","sumNonlinear_tr.nrrd"),"-M","-1",
           "-o",os.path.join("residualDir", "sumNonlinear_inv_tr.nrrd")]
call(command)

myfileImages = open("refIms.txt","w")
myfileMasks = open("masksIms.txt","w")
for a in range(1,args.num_images+1):
    if a == 1 and args.num_iter == 1:
        command = [animaTransformSerieXmlGenerator,"-i",os.path.join("residualDir", "sumNonlinear_inv_tr.nrrd"),
                   "-o",os.path.join("tempDir", "trsf_" + str(a) + ".xml")]
        call(command)
    else:
        command = [animaTransformSerieXmlGenerator,"-i",os.path.join("tempDir", args.prefix + "_" + str(a) + "_linear_tr.txt"),
                   "-i", os.path.join("tempDir",args.prefix + "_" + str(a) + "_nonlinear_tr.nrrd"),
                   "-i",os.path.join("residualDir","sumNonlinear_inv_tr.nrrd"),
                   "-o",os.path.join("tempDir","trsf_" + str(a) + ".xml")]
        call(command)

    command = [animaTensorApplyTransformSerie,"-i",os.path.join(args.prefix_base,args.prefix + "_" + str(a) + args.files_extension),
               "-t",os.path.join("tempDir","trsf_" + str(a) + ".xml"),"-g",args.ref_image,
               "-o",os.path.join("tempDir",args.prefix + "_" + str(a) + "_at.nrrd"),"-p",str(args.num_cores)]
    call(command)
    myfileImages.write(os.path.join("tempDir", args.prefix + "_" + str(a) + "_at.nrrd\n"))

    command = [animaComputeDTIScalarMaps,"-i",os.path.join("tempDir",args.prefix + "_" + str(a) + "_at.nrrd"),
               "-a",os.path.join("tempDir",args.prefix + "_" + str(a) + "_at_ADC.nrrd")]
    call(command)

    command = [animaThrImage,"-i",os.path.join("tempDir",args.prefix + "_" + str(a) + "_at_ADC.nrrd"),
               "-t","0","-o",os.path.join("tempDir","Mask_" + str(a) + "_at.nrrd")]
    call(command)
    myfileMasks.write(os.path.join("tempDir","Mask_" + str(a) + "_at.nrrd\n"))

myfileImages.close()
myfileMasks.close()

if args.num_iter == 0:
    command = [animaAverageImages,"-i","refIms.txt","-o","averageDTI1.nrrd","-m","masksIms.txt"]
else:
    command = [animaAverageImages,"-i","refIms.txt",
               "-o","averageDTI" + str(args.num_iter) + ".nrrd","-m","masksIms.txt"]

if not args.weights == "":
    command += ["-w",args.weights]
call(command)

command = [animaAverageImages,"-i","masksIms.txt","-o",os.path.join("tempDir","meanMasks_at.nrrd")]
if not args.weights == "":
    command += ["-w",args.weights]
call(command)

command = [animaThrImage,"-i",os.path.join("tempDir","meanMasks_at.nrrd"),"-t","0.25",
           "-o",os.path.join("tempDir","thrMeanMasks_at.nrrd")]
call(command)

if args.num_iter == 0:
    command = [animaMaskImage,"-i","averageDTI1.nrrd", "-m", os.path.join("tempDir", "thrMeanMasks_at.nrrd"),
               "-o", "averageDTI1.nrrd"]
    call(command)

    if os.path.exists("averageDTI1.nrrd"):
        open("it_1_done", "w").close()
        if os.path.exists("iterRun_2"):
            shutil.rmtree("residualDir")
            shutil.rmtree("tempDir")
            os.makedirs('tempDir')
            os.makedirs('residualDir')
            os.remove("iterRun_1")
else:
    command = [animaMaskImage,"-i","averageDTI" + str(args.num_iter) + ".nrrd",
               "-m",os.path.join("tempDir","thrMeanMasks_at.nrrd"),
               "-o","averageDTI" + str(args.num_iter) + ".nrrd"]
    call(command)

    if os.path.exists("averageDTI" + str(args.num_iter) + ".nrrd"):
        open("it_" + str(args.num_iter) + "_done","w").close()
        t = args.num_iter + 1
        if os.path.exists("iterRun_" + str(t)):
            shutil.rmtree("residualDir")
            shutil.rmtree("tempDir")
            os.makedirs('tempDir')
            os.makedirs('residualDir')
            os.remove("iterRun_" + str(args.num_iter))
