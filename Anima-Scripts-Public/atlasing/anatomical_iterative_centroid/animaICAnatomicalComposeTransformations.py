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
parser.add_argument('-a', '--num-img', type=int, required=True, help='Image number')
parser.add_argument('-b', '--bch-order', type=int, default=2, help='BCH order when composing transformations in rigid unbiased (default: 2)')
parser.add_argument('-i', '--num-iter', type=int, required=True, help='Iteration number of atlas creation')
parser.add_argument('-c', '--num-cores', type=int, default=40, help='Number of cores to run on')
parser.add_argument('-s', '--start', type=int, default=1, help='Number of images in the starting atlas (default: 1)')

args = parser.parse_args()
os.chdir(args.ref_dir)

k=args.num_iter
a=args.num_img

animaImageArithmetic = os.path.join(animaDir,"animaImageArithmetic")
animaTransformSerieXmlGenerator = os.path.join(animaDir,"animaTransformSerieXmlGenerator")
animaApplyTransformSerie = os.path.join(animaDir,"animaApplyTransformSerie")
animaDenseTransformArithmetic = os.path.join(animaDir,"animaDenseTransformArithmetic")
animaCreateImage = os.path.join(animaDir,"animaCreateImage")
animaLinearTransformArithmetic = os.path.join(animaDir,"animaLinearTransformArithmetic")

if a==1 and k==2:
    command = [animaCreateImage,"-g", "averageForm1.nii.gz", "-v", "3", "-b", "0", "-o", "tempDir/thetak_1.nii.gz"]
    call(command)
    command= [animaLinearTransformArithmetic, "-i", os.path.join("tempDir",args.prefix + "_2_linear_tr.txt"), "-M", "0", "-o", os.path.join("tempDir",args.prefix + "_1_linear_tr.txt")]
    call(command)

if a < k:
    command = [animaDenseTransformArithmetic,"-i",os.path.join("tempDir", "thetak_" + str(a) + ".nii.gz"), "-c", os.path.join("tempDir", "Tk.nii.gz"), "-b", str(args.bch_order), "-o", os.path.join("tempDir", "thetak_" + str(a) + ".nii.gz")]
    call(command)

command = [animaTransformSerieXmlGenerator,"-i", os.path.join("tempDir",args.prefix + "_" + str(a) + "_linear_tr.txt"), "-i", os.path.join("tempDir", "thetak_" + str(a) + ".nii.gz"), "-o", os.path.join("tempDir", "T_" + str(a) + ".xml") ]
call(command)

command = [animaApplyTransformSerie,"-i",os.path.join(args.prefix_base,args.prefix + "_" + str(a) + ".nii.gz"),"-t",os.path.join("tempDir", "T_" + str(a) + ".xml"),"-g","averageForm" + str(k-1) + ".nii.gz", "-o",os.path.join("tempDir",args.prefix + "_" + str(a) + "_at.nii.gz"),"-p",str(args.num_cores)]
call(command)

if os.path.exists(os.path.join("Masks", "Mask_" + str(a) + ".nii.gz")):
    command = [animaApplyTransformSerie,"-i",os.path.join("Masks", "Mask_" + str(a) + ".nii.gz"),"-t",os.path.join("tempDir", "T_" + str(a) + ".xml"),"-g","averageForm" + str(k-1) + ".nii.gz", "-o",os.path.join("tempDir","Mask_" + str(a) + "_at.nii.gz"),"-p",str(args.num_cores),"-n","nearest"]
    call(command)

   
