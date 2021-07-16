#!/usr/bin/python3

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
    description="Given an atlas built using the atlasing scripts of Anima, plus tracts start and end regions provided"
                "by tractseg, and MCM, computes fiber tracts on the DTI atlas and MCM atlas")
parser.add_argument('-n', '--num-subjects', type=int, required=True,
                    help="Number of subjects used for computing the atlas")

parser.add_argument('-a', '--dti-atlas-image', type=str, required=True, help='DTI atlas image')
parser.add_argument('-i', '--tensor-images-prefix', type=str, required=True, help='Tensor images prefix')
parser.add_argument('--mask-images-prefix', type=str, default='Preprocessed_DWI/DWI_BrainMask', help='Mask images prefix')
parser.add_argument('-m', '--mcm-images-prefix', type=str, required=True, help='MCM images prefix')
parser.add_argument('-t', '--tracts-folder', type=str, default='Tracts_Masks', help='Tract filter masks folder')

args = parser.parse_args()

# The goal here is to create, from an atlas and attached data coming from the atlasing script and MCM fiber preparation,
# the atlas of fibers itself. What is done here is:
# - Apply transforms to MCM, tracts masks
# - perform majority voting on each tract mask
# - Computes full brain tractography
# - Filters it to get atlas fiber tracts

animaComputeDTIScalarMaps = os.path.join(animaDir, "animaComputeDTIScalarMaps")
animaThrImage = os.path.join(animaDir, "animaThrImage")
animaTransformSerieXmlGenerator = os.path.join(animaDir, "animaTransformSerieXmlGenerator")
animaApplyTransformSerie = os.path.join(animaDir, "animaApplyTransformSerie")
animaMCMApplyTransformSerie = os.path.join(animaDir, "animaMCMApplyTransformSerie")
animaMCMAverageImages = os.path.join(animaDir, "animaMCMAverageImages")
animaAverageImages = os.path.join(animaDir, "animaAverageImages")
animaDTITractography = os.path.join(animaDir, "animaDTITractography")
animaMajorityLabelVoting = os.path.join(animaDir, "animaMajorityLabelVoting")
animaFibersFilterer = os.path.join(animaDir, "animaFibersFilterer")
animaTracksMCMPropertiesExtraction = os.path.join(animaDir, "animaTracksMCMPropertiesExtraction")

os.makedirs('Transformed_MCM', exist_ok=True)
os.makedirs('Transformed_Tracts_Masks', exist_ok=True)
os.makedirs('Atlas_Tracts', exist_ok=True)
os.makedirs('Augmented_Atlas_Tracts', exist_ok=True)

# Tracts list imported from tractseg
tracksLists = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6',
               'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'MLF_left', 'MLF_right', 'FPT_left', 'FPT_right',
               'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left', 'ILF_right', 'MCP',
               'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right', 'SLF_I_left', 'SLF_I_right',
               'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right', 'STR_left', 'STR_right', 'UF_left',
               'UF_right', 'CC', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left', 'T_PREM_right', 'T_PREC_left',
               'T_PREC_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PAR_left', 'T_PAR_right', 'T_OCC_left',
               'T_OCC_right', 'ST_FO_left', 'ST_FO_right', 'ST_PREF_left', 'ST_PREF_right', 'ST_PREM_left',
               'ST_PREM_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_POSTC_left', 'ST_POSTC_right', 'ST_PAR_left',
               'ST_PAR_right', 'ST_OCC_left', 'ST_OCC_right']

tensorsPrefixBase = os.path.dirname(args.tensor_images_prefix)
tensorsPrefix = os.path.basename(args.tensor_images_prefix)
mcmPrefixBase = os.path.dirname(args.mcm_images_prefix)
mcmPrefix = os.path.basename(args.mcm_images_prefix)
maskPrefixBase = os.path.dirname(args.mask_images_prefix)
maskPrefix = os.path.basename(args.mask_images_prefix)
mcmListFile = open(os.path.join('Transformed_MCM', 'listMCM.txt'), "w")
mcmB0ListFile = open(os.path.join('Transformed_MCM', 'listMCM_B0.txt'), "w")
mcmS2ListFile = open(os.path.join('Transformed_MCM', 'listMCM_S2.txt'), "w")
maskListFile = open(os.path.join('Transformed_MCM', 'listMasks.txt'), "w")

for dataNum in range(1, args.num_subjects + 1):
    # Apply transformations to additional MCM, assumes all transforms are in residualDir
    trsfGeneratorCommand = [animaTransformSerieXmlGenerator, "-i", os.path.join("residualDir", tensorsPrefix + "_" + str(dataNum) + "_linear_tr.txt"),
                            "-i", os.path.join("residualDir", tensorsPrefix + "_" + str(dataNum) + "_nonlinear_tr.nrrd"),
                            "-i", os.path.join("residualDir", "sumNonlinear_inv_tr.nrrd"), "-o", os.path.join("residualDir", "trsf_" + str(dataNum) + ".xml")]
    call(trsfGeneratorCommand)

    mcmApplyCommand = [animaMCMApplyTransformSerie,
                       "-i", os.path.join(mcmPrefixBase, mcmPrefix + "_" + str(dataNum) + ".mcm"),
                       "-o", os.path.join('Transformed_MCM', mcmPrefix + "_" + str(dataNum) + ".mcm"),
                       "-t", os.path.join("residualDir", "trsf_" + str(dataNum) + ".xml"),
                       "-g", args.dti_atlas_image, "-n", "3"]
    call(mcmApplyCommand)

    mcmListFile.write(os.path.join(os.getcwd(), 'Transformed_MCM', mcmPrefix + "_" + str(dataNum) + ".mcm") + "\n")

    mcmB0ApplyCommand = [animaApplyTransformSerie,
                         "-i", os.path.join(mcmPrefixBase, mcmPrefix + "_B0_" + str(dataNum) + ".nrrd"),
                         "-o", os.path.join('Transformed_MCM', mcmPrefix + "_B0_" + str(dataNum) + ".nrrd"),
                         "-t", os.path.join("residualDir", "trsf_" + str(dataNum) + ".xml"),
                         "-g", args.dti_atlas_image]
    call(mcmB0ApplyCommand)

    mcmB0ListFile.write(os.path.join(os.getcwd(), 'Transformed_MCM', mcmPrefix + "_B0_" + str(dataNum) + ".nrrd") + "\n")

    mcmS2ApplyCommand = [animaApplyTransformSerie,
                         "-i", os.path.join(mcmPrefixBase, mcmPrefix + "_S2_" + str(dataNum) + ".nrrd"),
                         "-o", os.path.join('Transformed_MCM', mcmPrefix + "_S2_" + str(dataNum) + ".nrrd"),
                         "-t", os.path.join("residualDir", "trsf_" + str(dataNum) + ".xml"),
                         "-g", args.dti_atlas_image]
    call(mcmS2ApplyCommand)

    mcmS2ListFile.write(os.path.join(os.getcwd(), 'Transformed_MCM', mcmPrefix + "_S2_" + str(dataNum) + ".nrrd") + "\n")

    maskApplyCommand = [animaApplyTransformSerie,
                        "-i", os.path.join(maskPrefixBase, maskPrefix + "_" + str(dataNum) + ".nrrd"),
                        "-o", os.path.join('Transformed_MCM', maskPrefix + "_" + str(dataNum) + ".nrrd"),
                        "-t", os.path.join("residualDir", "trsf_" + str(dataNum) + ".xml"),
                        "-g", args.dti_atlas_image, "-n", "nearest"]
    call(maskApplyCommand)

    maskListFile.write(os.path.join(os.getcwd(), 'Transformed_MCM', maskPrefix + "_" + str(dataNum) + ".nrrd") + "\n")

    # Now apply the transform to all tractseg regions
    for track in tracksLists:
        # Apply transform to fused begin and end mask
        applyCommand = [animaApplyTransformSerie,
                        "-i", os.path.join(args.tracts_folder, track + "_" + str(dataNum) + ".nrrd"),
                        "-o", os.path.join('Transformed_Tracts_Masks', track + "_" + str(dataNum) + ".nrrd"),
                        "-t", os.path.join("residualDir", "trsf_" + str(dataNum) + ".xml"),
                        "-g", args.dti_atlas_image, "-n", "nearest"]
        call(applyCommand)

# Main loop done, now perform averaging of MCM
mcmListFile.close()
mcmB0ListFile.close()
mcmS2ListFile.close()
maskListFile.close()

mergeMCMCommand = [animaMCMAverageImages, "-i", os.path.join('Transformed_MCM', 'listMCM.txt'), "-n", "3",
                   "-m", os.path.join('Transformed_MCM', 'listMasks.txt'), "-o", "averageMCM.mcm"]
call(mergeMCMCommand)

mergeMCMB0Command = [animaAverageImages, "-i", os.path.join('Transformed_MCM', 'listMCM_B0.txt'), "-m", os.path.join('Transformed_MCM', 'listMasks.txt'), "-o", "averageMCM_B0.nrrd"]
call(mergeMCMB0Command)

mergeMCMS2Command = [animaAverageImages, "-i", os.path.join('Transformed_MCM', 'listMCM_S2.txt'), "-m", os.path.join('Transformed_MCM', 'listMasks.txt'), "-o", "averageMCM_S2.nrrd"]
call(mergeMCMS2Command)

# Perform tractography on average MCM model
adcCommand = [animaComputeDTIScalarMaps, "-i", args.dti_atlas_image, "-a", "averageADC.nrrd"]
call(adcCommand)

thrCommand = [animaThrImage, "-t", "0", "-i", "averageADC.nrrd", "-o",
              "averageMask.nrrd"]
call(thrCommand)

trackingCommand = [animaDTITractography, "-i", args.dti_atlas_image, "-s", "averageMask.nrrd", "--nb-fibers", "2", "-a", "90", "-p", "0",
                   "-o", os.path.join('Atlas_Tracts', 'WholeBrain_Tractography.fds')]
call(trackingCommand)

# Majority vote for tracts masks and filter main tractography
for track in tracksLists:
    trackMasksListFile = open(os.path.join('Transformed_Tracts_Masks', 'listMasks.txt'), "w")
    for dataNum in range(1, args.num_subjects + 1):
        trackMasksListFile.write(os.path.join(os.getcwd(), 'Transformed_Tracts_Masks', track + "_" + str(dataNum) + ".nrrd") + "\n")

    trackMasksListFile.close()

    majorityVoteCommand = [animaMajorityLabelVoting, "-i", os.path.join('Transformed_Tracts_Masks', 'listMasks.txt'),
                           "-o", os.path.join('Transformed_Tracts_Masks', track + '_FilterMask.nrrd')]
    call(majorityVoteCommand)

    fiberFilterCommand = [animaFibersFilterer, "-i", os.path.join('Atlas_Tracts', 'WholeBrain_Tractography.fds'),
                          "-o", os.path.join('Atlas_Tracts', track + '.fds'),
                          "-r", os.path.join('Transformed_Tracts_Masks', track + '_FilterMask.nrrd'),
                          "-e", "1", "-e", "2"]
    call(fiberFilterCommand)

    trackListFile = open(os.path.join('Augmented_Atlas_Tracts', 'listData_' + track + '.txt'), "w")
    for dataNum in range(1, args.num_subjects + 1):
        propsExtractionCommand = [animaTracksMCMPropertiesExtraction, "-i", os.path.join('Atlas_Tracts', track + '.fds'),
                                  "-m", os.path.join('Transformed_MCM', mcmPrefix + "_" + str(dataNum) + ".mcm"),
                                  "-o", os.path.join('Augmented_Atlas_Tracts', track + '_MCM_augmented_' + str(dataNum) + '.fds')]
        call(propsExtractionCommand)

        trackListFile.write(os.path.join(os.getcwd(), 'Augmented_Atlas_Tracts', track + '_MCM_augmented_' + str(dataNum) + '.fds') + "\n")

    trackListFile.close()

