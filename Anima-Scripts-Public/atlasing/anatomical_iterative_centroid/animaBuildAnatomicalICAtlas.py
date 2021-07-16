#!/usr/bin/python3
# Warning: works only on unix-like systems, not windows where "python animaBuildAnatomicalAtlas.py ..." has to be run

import argparse
import os
import glob
import stat
import sys
import subprocess
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
animaScriptsDir = configParser.get("anima-scripts", 'anima-scripts-public-root')

# Argument parsing
parser = argparse.ArgumentParser(
    description="Builds and runs a series of scripts on an OAR cluster to construct an anatomical atlas (unbiased up to an affine or rigid transform, with different or equal weights).")
parser.add_argument('-p', '--data-prefix', type=str, required=True, help='Data prefix (including folder)')
parser.add_argument('-n', '--num-images', type=int, required=True, help='Number of images in the atlas')
parser.add_argument('-c', '--num-cores', type=int, default=8, help='Number of cores to run on (default: 8)')
parser.add_argument('-b', '--bch-order', type=int, default=2, help='BCH order when composing transformations (default: 2)')
parser.add_argument('-s', '--start', type=int, default=1, help='number of images in the starting atlas (default: 1)')
parser.add_argument('--rigid', action='store_true', help="Unbiased atlas up to a rigid transformation")

args = parser.parse_args()

if not os.path.exists('tempDir'):
    os.makedirs('tempDir')

if os.path.exists('residualDir'):
    shutil.rmtree("residualDir")

os.makedirs('residualDir')

prefixBase = os.path.dirname(args.data_prefix)
prefix = os.path.basename(args.data_prefix)

if args.start == 1 or args.start == 0:
    shutil.copyfile(os.path.join(prefixBase, prefix + "_1.nii.gz"), "averageForm1.nii.gz")
    args.start = 1 

previousMergeId = 0

for k in range(args.start + 1, args.num_images + 1):
    if os.path.exists('it_' + str(k) + '_done'):
        firstImage = 1
        continue
    ref = "averageForm" + str(k-1)

    print("*************Incorporating image: " + str(k) + " in atlas: " + ref)

    for f in glob.glob("residualDir/" + prefix + '_*_nl_tr.nii.gz') + glob.glob("residualDir/" + prefix + '_*_flag'):
        os.remove(f)

    nCoresPhysical = int(args.num_cores / 2)

    fileName = 'regRun_' + str(k)
    myfile = open(fileName,"w")
    myfile.write("#!/bin/bash\n")
    if args.num_cores<=16:
        myfile.write("#OAR -l {hyperthreading=\'NO\'}/nodes=1/core=" + str(args.num_cores) + ",walltime=01:59:00\n")
    myfile.write("#OAR -l {hyperthreading=\'YES\'}/nodes=1/core=" + str(nCoresPhysical) + ",walltime=01:59:00\n")
    myfile.write("#OAR -O " + os.getcwd() + "/reg-" + str(k) + ".%jobid%.output\n")
    myfile.write("#OAR -E " + os.getcwd() + "/reg-" + str(k) + ".%jobid%.error\n")

    myfile.write("cd " + os.getcwd() + "\n")

    myfile.write(os.path.join(animaScriptsDir,"atlasing/anatomical_iterative_centroid/animaICAnatomicalRegisterImage.py") +
                 " -d " + os.getcwd() + " -r " + ref + ".nii.gz -B " + prefixBase + " -p " + prefix + " -i " + str(k) +
                 " -b " + str(args.bch_order) + " -c " + str(args.num_cores))

    if args.rigid is True:
        myfile.write(" --rigid\n")
    else:
        myfile.write("\n")

    myfile.close()
    os.chmod(fileName, stat.S_IRWXU)

    oarRunCommand = ["oarsub"]
    if previousMergeId == 0:
        oarRunCommand += ["-n","reg-" + str(k),"-S", os.getcwd() + "/regRun_" + str(k)]
    else:
        oarRunCommand += ["-n","reg-" + str(k),"-a",str(previousMergeId),"-S", os.getcwd() + "/regRun_" + str(k)]

    procStat = subprocess.run(oarRunCommand, stdout=subprocess.PIPE)
    statLines = procStat.stdout.decode('utf-8').split('\n')
    for statsLine in statLines:
        if "OAR_JOB_ID" in statsLine:
            previousRegId = statsLine.split("=")[1]
            break

    numJobs = k

    fileName = 'bchRun_' + str(k)
    myfile = open(fileName,"w")
    myfile.write("#!/bin/bash\n")
    if args.num_cores<=16:
        myfile.write("#OAR -l {hyperthreading=\'NO\'}/nodes=1/core=" + str(args.num_cores) + ",walltime=01:59:00\n")
    myfile.write("#OAR -l {hyperthreading=\'YES\'}/nodes=1/core=" + str(nCoresPhysical) + ",walltime=01:59:00\n")
    myfile.write("#OAR --array " + str(numJobs) + "\n")
    myfile.write("#OAR -O " + os.getcwd() + "/bch-" + str(k) + ".%jobid%.output\n")
    myfile.write("#OAR -E " + os.getcwd() + "/bch-" + str(k) + ".%jobid%.error\n")

    myfile.write("cd " + os.getcwd() + "\n")

    myfile.write(os.path.join(animaScriptsDir,"atlasing/anatomical_iterative_centroid/animaICAnatomicalComposeTransformations.py") +
                 " -d " + os.getcwd() + " -B " + prefixBase + " -p " + prefix + " -i " + str(k) +
                 " -c " + str(args.num_cores) + " -s " + str(args.start) + " -b " + str(args.bch_order) +
                 " -a $OAR_ARRAY_INDEX \n")

    myfile.close()
    os.chmod(fileName, stat.S_IRWXU)

    oarRunCommand = ["oarsub","-n","bch-" + str(k),"-S",os.getcwd() + "/bchRun_" + str(k), "-a", str(previousRegId)]
    
    jobsIds = []
    procStat = subprocess.run(oarRunCommand, stdout=subprocess.PIPE)
    statLines = procStat.stdout.decode('utf-8').split('\n')
    for statsLine in statLines:
        if "OAR_JOB_ID" in statsLine:
            jobsIds += [statsLine.split("=")[1]]

    fileName = 'mergeRun_' + str(k)
    myfile = open(fileName,"w")
    myfile.write("#!/bin/bash\n")
    if args.num_cores<=16:
        myfile.write("#OAR -l {hyperthreading=\'NO\'}/nodes=1/core=" + str(args.num_cores) + ",walltime=01:59:00\n")
    myfile.write("#OAR -l {hyperthreading=\'YES\'}/nodes=1/core=" + str(nCoresPhysical) + ",walltime=01:59:00\n")
    myfile.write("#OAR -O " + os.getcwd() + "/merge-" + str(k) + ".%jobid%.output\n")
    myfile.write("#OAR -E " + os.getcwd() + "/merge-" + str(k) + ".%jobid%.error\n")

    myfile.write("cd " + os.getcwd() + "\n")

    myfile.write(os.path.join(animaScriptsDir,"atlasing/anatomical_iterative_centroid/animaICAnatomicalMergeImages.py") +
                 " -d " + os.getcwd() + " -B " + prefixBase + " -p " + prefix + " -i " + str(k) +
                 " -c " + str(args.num_cores) + "\n")

    myfile.close()
    os.chmod(fileName, stat.S_IRWXU)

    oarRunCommand = ["oarsub"]
    for jobId in jobsIds:
        oarRunCommand += ["-a",jobId]
    oarRunCommand += ["-n","merge-" + str(k),"-S", os.getcwd() + "/mergeRun_" + str(k)]

    procStat = subprocess.run(oarRunCommand, stdout=subprocess.PIPE)
    statLines = procStat.stdout.decode('utf-8').split('\n')
    for statsLine in statLines:
        if "OAR_JOB_ID" in statsLine:
            previousMergeId = statsLine.split("=")[1]
            break
