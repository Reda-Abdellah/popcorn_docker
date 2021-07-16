#!/usr/bin/python3
# Warning: works only on unix-like systems, not windows where "python animaAtlasBasedBrainExtraction.py ..." has to be run

import sys
import argparse
import os
import stat
import subprocess

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
animaPyramidalBMRegistration = os.path.join(animaDir, "animaPyramidalBMRegistration")
animaDenseSVFBMRegistration = os.path.join(animaDir, "animaDenseSVFBMRegistration")
animaTransformSerieXmlGenerator = os.path.join(animaDir, "animaTransformSerieXmlGenerator")
animaApplyTransformSerie = os.path.join(animaDir, "animaApplyTransformSerie")
animaConvertImage = os.path.join(animaDir, "animaConvertImage")
animaConcatenateImages = os.path.join(animaDir, "animaConcatenateImages")
animaMajorityLabelVoting = os.path.join(animaDir, "animaMajorityLabelVoting")

# Argument parsing
parser = argparse.ArgumentParser(description="Propagate and fuse segmentations from multiple atlases onto a list of subjects")

parser.add_argument('-i', '--image-file', required=True, type=str, help='list of anatomical images to be segmented (in txt file)')
parser.add_argument('-a', '--anat-file', required=True, type=str, help='list of atlas anatomical images (in txt file)')
parser.add_argument('-s', '--seg-file', required=True, type=str, help='list of atlas label images (segmentations) (in txt file)')
parser.add_argument('-o', '--out-dir', required=True, type=str, help='output directory')
parser.add_argument('-c', '--num-cores', type=int, default=8, help='Number of cores to run on (default: 8)')

args = parser.parse_args()

images = [line.rstrip('\n') for line in open(args.image_file)]            
N = len(images)
anats = [line.rstrip('\n') for line in open(args.anat_file)]
P = len(anats)
segs = [line.rstrip('\n') for line in open(args.seg_file)]

if len(segs) != P:
    print("The number of atlas anatomical images must be equal to the number of atlas label images")
    exit()

outDir = args.out_dir
if not os.path.exists(outDir):
    os.makedirs(outDir)
if not os.path.exists(os.path.join(outDir, "err")):
    os.makedirs(os.path.join(outDir, "err"))
if not os.path.exists(os.path.join(outDir, "out")):
    os.makedirs(os.path.join(outDir, "out"))
if not os.path.exists(os.path.join(outDir, "registrations")):
    os.makedirs(os.path.join(outDir, "registrations"))
if not os.path.exists(os.path.join(outDir, "segmentations")):
    os.makedirs(os.path.join(outDir, "segmentations"))
    
for i in range(0, N):
    image=images[i]  
    
    # Registrations
    
    imagePrefix = os.path.splitext(image)[0]
    if os.path.splitext(image)[1] == '.gz':
        imagePrefix = os.path.splitext(imagePrefix)[0]
    imageBasename = os.path.basename(imagePrefix)

    nCoresPhysical = int(args.num_cores / 2)
        
    filename = os.path.join(outDir, "regRun_" + imageBasename)
    myfile = open(filename,"w")
    myfile.write("#!/bin/bash\n")
    if args.num_cores<=16:
        myfile.write("#OAR -l {hyperthreading=\'NO\'}/nodes=1/core=" + str(args.num_cores) + ",walltime=01:59:00\n")
    myfile.write("#OAR -l {hyperthreading=\'YES\'}/nodes=1/core=" + str(nCoresPhysical) + ",walltime=01:59:00\n")
    myfile.write("#OAR --array " + str(P) + "\n")
    myfile.write("#OAR -O " + os.path.join(outDir, "out" , imageBasename) + ".%jobid%.output\n")
    myfile.write("#OAR -E " + os.path.join(outDir, "err" , imageBasename) + ".%jobid%.error\n")
    myfile.write("anats=(" + " ".join(anats) + ")\n")
    myfile.write("segs=(" + " ".join(segs) + ")\n")            
    myfile.write(animaPyramidalBMRegistration + " -m ${anats[$(($OAR_ARRAY_INDEX-1))]} -r " + image + " -o " + os.path.join(outDir, "registrations", imageBasename) + "_${OAR_ARRAY_INDEX}_aff.nrrd -O " + os.path.join(outDir, "registrations", imageBasename) + "_${OAR_ARRAY_INDEX}_aff_tr.txt --sp 3 --ot 2 -p 4 -l 0" + "\n" )
    myfile.write(animaDenseSVFBMRegistration + " -m " + os.path.join(outDir, "registrations", imageBasename) + "_${OAR_ARRAY_INDEX}_aff.nrrd -r " + image + " -o " + os.path.join(outDir, "registrations", imageBasename) + "_${OAR_ARRAY_INDEX}_diffeo.nrrd -O " + os.path.join(outDir, "registrations", imageBasename) + "_${OAR_ARRAY_INDEX}_diffeo_tr.nrrd --sr 1 -p 3 -l 0" + "\n" )
    myfile.write(animaTransformSerieXmlGenerator + " -i " + os.path.join(outDir, "registrations", imageBasename) + "_${OAR_ARRAY_INDEX}_aff_tr.txt -i " + os.path.join(outDir, "registrations", imageBasename) + "_${OAR_ARRAY_INDEX}_diffeo_tr.nrrd -o " + os.path.join(outDir, "registrations", imageBasename) + "_${OAR_ARRAY_INDEX}_tr.xml\n" )
    myfile.write(animaApplyTransformSerie + " -i ${segs[$(($OAR_ARRAY_INDEX-1))]} -g " + image + " -t " + os.path.join(outDir, "registrations", imageBasename) + "_${OAR_ARRAY_INDEX}_tr.xml" + " -o " + os.path.join(outDir, "segmentations", imageBasename) + "_${OAR_ARRAY_INDEX}_seg.nrrd -n nearest\n" )
    myfile.close()

    os.chmod(filename, stat.S_IRWXU)
    oarRunCommand = ["oarsub","-n","reg-" + imageBasename,"-S", filename]
    jobsIds = []
    procStat = subprocess.run(oarRunCommand, stdout=subprocess.PIPE)
    statLines = procStat.stdout.decode('utf-8').split('\n')
    for statsLine in statLines:
        if "OAR_JOB_ID" in statsLine:
            jobsIds += [statsLine.split("=")[1]]
    
    # Label fusion
    
    listSeg=os.path.join(outDir, "segmentations", imageBasename) + "_listSeg.txt"
    segFile = open(listSeg,"w")
    for j in range(0, P):
        segFile.write(os.path.join(outDir, "segmentations", imageBasename) + "_" + str(j+1) + "_seg.nrrd\n")
    segFile.close()

    filename2 = os.path.join(outDir, "fuseRun_" + imageBasename)
    myfile2 = open(filename2,"w")
    myfile2.write("#!/bin/bash\n")
    if args.num_cores<=16:
        myfile2.write("#OAR -l {hyperthreading=\'NO\'}/nodes=1/core=" + str(args.num_cores) + ",walltime=01:59:00\n")
    myfile2.write("#OAR -l {hyperthreading=\'YES\'}/nodes=1/core=" + str(nCoresPhysical) + ",walltime=01:59:00\n")
    myfile2.write("#OAR -O " + os.path.join(outDir, "out" , imageBasename) + "_fusion.%jobid%.output\n")
    myfile2.write("#OAR -E " + os.path.join(outDir, "err" , imageBasename) + "_fusion.%jobid%.error\n")
    myfile2.write(animaMajorityLabelVoting + " -i " + listSeg + " -o " + os.path.join(outDir, imageBasename) + "_consensus_seg.nrrd \n")
    myfile2.close()
    
    os.chmod(filename2, stat.S_IRWXU)
    oarFuseCommand = ["oarsub","-n","fusion_" + imageBasename,"-S", filename2]            

    for jobId in jobsIds:
        oarFuseCommand += ["-a",jobId]
    
    subprocess.call(oarFuseCommand, stdout=open(os.devnull, "w"))
