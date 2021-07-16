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
parser.add_argument('-B', '--prefix-base', type=str, required=True, help='Prefix base')
parser.add_argument('-p', '--prefix', type=str, required=True, help='Prefix')
parser.add_argument('-i', '--num-iter', type=int, required=True, help='Iteration number of atlas creation')
parser.add_argument('-c', '--num-cores', type=int, default=40, help='Number of cores to run on')

args = parser.parse_args()
os.chdir(args.ref_dir)

animaAverageImages = os.path.join(animaDir,"animaAverageImages")

myfile = open("avgImg.txt","w")
myfileMasks = open("masksIms.txt","w")
for a in range(1,args.num_iter + 1):
    myfile.write(os.path.join("tempDir",args.prefix + "_" + str(a) + "_at.nii.gz") + "\n")

    if os.path.exists(os.path.join("Masks", "Mask_" + str(a) + ".nii.gz")):
        myfileMasks.write(os.path.join("tempDir","Mask_" + str(a) + "_at.nii.gz\n"))

myfile.close()
myfileMasks.close()

command = [animaAverageImages, "-i", "avgImg.txt","-o","averageForm" + str(args.num_iter) +".nii.gz"]

if os.path.exists(os.path.join("Masks","Mask_1.nii.gz")):
    command += ["-m","masksIms.txt"]

call(command)
