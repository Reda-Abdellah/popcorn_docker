#!/usr/bin/python3

import sys
import argparse

if sys.version_info[0] > 2:
    import configparser as ConfParser
else:
    import ConfigParser as ConfParser

import tempfile
import os
import glob
from subprocess import call
import shutil

configFilePath = os.path.join(os.path.expanduser("~"), ".anima",  "config.txt")
if not os.path.exists(configFilePath):
    print('Please create a configuration file for Anima python scripts. Refer to the README')
    quit()

configParser = ConfParser.RawConfigParser()
configParser.read(configFilePath)

animaDir = configParser.get("anima-scripts", 'anima')
animaScriptsDir = configParser.get("anima-scripts", 'anima-scripts-public-root')

# Argument parsing
parser = argparse.ArgumentParser(description="Given a fiber atlas constructed from controls data, and a patient image, performs patient to atlas comparison")
parser.add_argument('-n', '--num-subjects', type=int, required=True,
                    help="Number of subjects used for computing the atlas")

parser.add_argument('-i', '--dw-patient-image', type=str, required=True, help='DW patient image (folder + name)')
parser.add_argument('-d', '--dw-dicom-folder', type=str, default="", help='Dicom folder for patient')
parser.add_argument('-t', '--t1-image', type=str, required=True, help='T1 patient image (folder + name)')
parser.add_argument('--dw-without-reversed-b0', action='store_true', help="No reversed B0 provided with the patient DWI, otherwise assume there is a file named by DWI prefix + reverse_b0.nii.gz")
parser.add_argument('--type', type=str, default="tensor", help="Type of compartment model for fascicles (stick, zeppelin, tensor, noddi, ddi)")
parser.add_argument('--register-t1-on-dwi', action='store_true', help="T1 registration on DWI is needed as they were not acquired in the same session")

parser.add_argument('-a', '--dti-atlas-image', type=str, required=True, help='DTI atlas image')
parser.add_argument('-r', '--raw-tracts-folder', type=str, default='Atlas_Tracts', help='Raw atlas tracts folder')
parser.add_argument('--tracts-folder', type=str, default='Augmented_Atlas_Tracts', help='Atlas tracts augmented with controls data')

args = parser.parse_args()

# The goal here is to create, from an atlas and patient's data, stats along tracts of differences. The script should preferably be run in the atlas folder.
# What is done here is:
# - pre-process patient image as in subjects preparation
# - register DTI patient image on atlas DTI image, apply transformation to MCM image
# - for each tract, augment raw atlas tracts with data from patient
# - for each tract, compute pairwise tests along tracts between patient and controls

animaComputeDTIScalarMaps = os.path.join(animaDir, "animaComputeDTIScalarMaps")
animaPyramidalBMRegistration = os.path.join(animaDir, "animaPyramidalBMRegistration")
animaTransformSerieXmlGenerator = os.path.join(animaDir, "animaTransformSerieXmlGenerator")
animaCreateImage = os.path.join(animaDir, "animaCreateImage")
animaApplyTransformSerie = os.path.join(animaDir, "animaApplyTransformSerie")
animaMaskImage = os.path.join(animaDir, "animaMaskImage")
animaTensorApplyTransformSerie = os.path.join(animaDir, "animaTensorApplyTransformSerie")
animaDenseTensorSVFBMRegistration = os.path.join(animaDir, "animaDenseTensorSVFBMRegistration")
animaMCMApplyTransformSerie = os.path.join(animaDir, "animaMCMApplyTransformSerie")
animaTracksMCMPropertiesExtraction = os.path.join(animaDir, "animaTracksMCMPropertiesExtraction")
animaPatientToGroupComparisonOnTracks = os.path.join(animaDir, "animaPatientToGroupComparisonOnTracks")
animaFibersApplyTransformSerie = os.path.join(animaDir, "animaFibersApplyTransformSerie")
animaFibersFDRCorrectPValues = os.path.join(animaDir, "animaFibersFDRCorrectPValues")
animaFibersDiseaseScores = os.path.join(animaDir, "animaFibersDiseaseScores")

os.makedirs('Preprocessed_Patients_DWI', exist_ok=True)
os.makedirs('Patients_Tensors', exist_ok=True)
os.makedirs('Patients_MCM', exist_ok=True)
os.makedirs('Transformed_Patients_MCM', exist_ok=True)
os.makedirs('Patients_Augmented_Tracts', exist_ok=True)
os.makedirs('Patients_Disease_Scores', exist_ok=True)

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

dwiPrefixBase = os.path.dirname(args.dw_patient_image)
dwiBasename = os.path.basename(args.dw_patient_image)
dwiPrefix = os.path.splitext(dwiBasename)[0]
if os.path.splitext(dwiBasename)[1] == '.gz':
    dwiPrefix = os.path.splitext(dwiPrefix)[0]

# Preprocess patient diffusion data
preprocCommand = ["python3", os.path.join(animaScriptsDir, "diffusion", "animaDiffusionImagePreprocessing.py"), "-b", os.path.join(dwiPrefixBase, dwiPrefix + ".bval"),
                  "-t", args.t1_image, "-i", args.dw_patient_image]

if args.register_t1_on_dwi is True:
    preprocCommand += ["--register-t1-on-dwi"]

if not args.dw_without_reversed_b0:
    preprocCommand += ["-r", os.path.join(dwiPrefixBase, dwiPrefix + "_reversed_b0.nii.gz")]

if args.dw_dicom_folder == "":
    preprocCommand += ["-g", os.path.join(dwiPrefixBase, dwiPrefix + ".bvec")]
else:
    dicomGlobFiles = glob.glob(os.path.join(args.dw_dicom_folder, "*"))
    preprocCommand += ["-D"] + dicomGlobFiles

call(preprocCommand)

# Move preprocessed results to output folders
shutil.move(os.path.join(dwiPrefixBase, dwiPrefix + "_Tensors.nrrd"), os.path.join("Patients_Tensors", dwiPrefix + "_Tensors.nrrd"))
shutil.move(os.path.join(dwiPrefixBase, dwiPrefix + "_preprocessed.bvec"), os.path.join("Preprocessed_Patients_DWI", dwiPrefix + ".bvec"))
shutil.copy(os.path.join(dwiPrefixBase, dwiPrefix + ".bval"), os.path.join("Preprocessed_Patients_DWI", dwiPrefix + ".bval"))
shutil.move(os.path.join(dwiPrefixBase, dwiPrefix + "_preprocessed.nrrd"), os.path.join("Preprocessed_Patients_DWI", dwiPrefix + ".nrrd"))
shutil.move(os.path.join(dwiPrefixBase, dwiPrefix + "_brainMask.nrrd"), os.path.join("Preprocessed_Patients_DWI", dwiPrefix + "_BrainMask.nrrd"))
os.remove(os.path.join(dwiPrefixBase, dwiPrefix + "_Tensors_B0.nrrd"))
os.remove(os.path.join(dwiPrefixBase, dwiPrefix + "_Tensors_NoiseVariance.nrrd"))

# Now estimate MCMs
os.chdir("Preprocessed_Patients_DWI")
mcmCommand = ["python3", os.path.join(animaScriptsDir, "diffusion", "animaMultiCompartmentModelEstimation.py"), "-i", dwiPrefix + ".nrrd",
              "-g", dwiPrefix + ".bvec", "-b", dwiPrefix + ".bval", "-n", "3", "-m", dwiPrefix + "_BrainMask.nrrd",
              "-t", args.type]
call(mcmCommand)

# Now move results to MCM folder
os.chdir("..")
shutil.move(os.path.join("Preprocessed_Patients_DWI", dwiPrefix + "_MCM_avg.mcm"), os.path.join("Patients_MCM", dwiPrefix + "_MCM_avg.mcm"))

if os.path.exists(os.path.join("Patients_MCM", dwiPrefix + "_MCM_avg")):
    shutil.rmtree(os.path.join("Patients_MCM", dwiPrefix + "_MCM_avg"), ignore_errors=True)
shutil.move(os.path.join("Preprocessed_Patients_DWI", dwiPrefix + "_MCM_avg"), os.path.join("Patients_MCM", dwiPrefix + "_MCM_avg"))

shutil.move(os.path.join("Preprocessed_Patients_DWI", dwiPrefix + "_MCM_B0_avg.nrrd"), os.path.join("Patients_MCM", dwiPrefix + "_MCM_avg_B0.nrrd"))
shutil.move(os.path.join("Preprocessed_Patients_DWI", dwiPrefix + "_MCM_S2_avg.nrrd"), os.path.join("Patients_MCM", dwiPrefix + "_MCM_avg_S2.nrrd"))

for f in glob.glob(os.path.join("Preprocessed_Patients_DWI", dwiPrefix + "_MCM*")):
    if os.path.isdir(f):
        shutil.rmtree(f, ignore_errors=True)
    else:
        os.remove(f)

# Register patient onto atlas (same as register DT Image in atlas construction)
tmpFolder = tempfile.mkdtemp()
adcCommand = [animaComputeDTIScalarMaps, "-i", args.dti_atlas_image, "-a", os.path.join(tmpFolder, "averageADC.nrrd")]
call(adcCommand)

adcCommand = [animaComputeDTIScalarMaps, "-i", os.path.join("Patients_Tensors", dwiPrefix + "_Tensors.nrrd"),
              "-a", os.path.join("Patients_Tensors", dwiPrefix + "_ADC.nrrd")]
call(adcCommand)

regCommand = [animaPyramidalBMRegistration, "-r", os.path.join(tmpFolder, "averageADC.nrrd"), "-m", os.path.join("Patients_Tensors", dwiPrefix + "_ADC.nrrd"),
              "-o", os.path.join(tmpFolder, "Patient_aff.nrrd"), "-O", os.path.join(tmpFolder, "Patient_aff_tr.txt"),
              "--ot", "2", "-p", "3", "-l", "0", "-I", "2", "--sym-reg", "2", "-s", "0"]
call(regCommand)

command = [animaTransformSerieXmlGenerator, "-i", os.path.join(tmpFolder, "Patient_aff_tr.txt"), "-o", os.path.join(tmpFolder, "Patient_aff_tr.xml")]
call(command)

command = [animaCreateImage, "-b", "1", "-v", "1", "-g", os.path.join("Patients_Tensors", dwiPrefix + "_ADC.nrrd"),
           "-o", os.path.join(tmpFolder,"tmpFullMask.nrrd")]
call(command)

command = [animaApplyTransformSerie, "-g", os.path.join(tmpFolder, "averageADC.nrrd"), "-i", os.path.join(tmpFolder,"tmpFullMask.nrrd"),
           "-t", os.path.join(tmpFolder, "Patient_aff_tr.xml"), "-o", os.path.join(tmpFolder,"tmpMask_aff.nrrd"),
           "-n", "nearest"]
call(command)

command = [animaMaskImage, "-i", args.dti_atlas_image, "-m", os.path.join(tmpFolder, "tmpMask_aff.nrrd"),
           "-o", os.path.join(tmpFolder, "refDTI_c.nrrd")]
call(command)

command = [animaTensorApplyTransformSerie, "-i", os.path.join("Patients_Tensors", dwiPrefix + "_Tensors.nrrd"),
           "-g", args.dti_atlas_image, "-t", os.path.join(tmpFolder, "Patient_aff_tr.xml"),
           "-o", os.path.join(tmpFolder, dwiPrefix + "_Tensors_aff.nrrd")]
call(command)

command = [animaDenseTensorSVFBMRegistration, "-r", os.path.join(tmpFolder, "refDTI_c.nrrd"),
           "-m", os.path.join(tmpFolder, dwiPrefix + "_Tensors_aff.nrrd"), "-o", os.path.join(tmpFolder, dwiPrefix + "_nl.nrrd"),
           "-O", os.path.join(tmpFolder, dwiPrefix + "_nl_tr.nrrd"), "--sr", "1", "--es", "3", "--fs", "2", "--sym-reg", "2", "--metric", "3", "-s", "0.001"]
call(command)

# Non linear registration done. Now applying to MCM image
command = [animaTransformSerieXmlGenerator, "-i", os.path.join(tmpFolder, "Patient_aff_tr.txt"), "-i", os.path.join(tmpFolder, dwiPrefix + "_nl_tr.nrrd"),
           "-o", os.path.join(tmpFolder, "Patient_nl_tr.xml")]
call(command)

mcmApplyCommand = [animaMCMApplyTransformSerie, "-i", os.path.join("Patients_MCM", dwiPrefix + "_MCM_avg.mcm"),
                   "-o", os.path.join('Transformed_Patients_MCM', dwiPrefix + "_MCM_avg_onAtlas.mcm"),
                   "-t", os.path.join(tmpFolder, "Patient_nl_tr.xml"), "-g", args.dti_atlas_image, "-n", "3"]
call(mcmApplyCommand)

mcmB0ApplyCommand = [animaApplyTransformSerie, "-i", os.path.join("Patients_MCM", dwiPrefix + "_MCM_avg_B0.nrrd"),
                     "-o", os.path.join('Transformed_Patients_MCM', dwiPrefix + "_MCM_avg_B0_onAtlas.nrrd"),
                     "-t", os.path.join(tmpFolder, "Patient_nl_tr.xml"), "-g", args.dti_atlas_image]
call(mcmB0ApplyCommand)

mcmS2ApplyCommand = [animaApplyTransformSerie, "-i", os.path.join("Patients_MCM", dwiPrefix + "_MCM_avg_S2.nrrd"),
                     "-o", os.path.join('Transformed_Patients_MCM', dwiPrefix + "_MCM_avg_S2_onAtlas.nrrd"),
                     "-t", os.path.join(tmpFolder, "Patient_nl_tr.xml"), "-g", args.dti_atlas_image]
call(mcmS2ApplyCommand)

# Align T1 reference image and DWI (not acquired in the same session
T1Prefix = os.path.splitext(args.t1_image)[0]
if os.path.splitext(args.t1_image)[1] == '.gz':
    T1Prefix = os.path.splitext(T1Prefix)[0]

t1RegistrationCommand = [animaPyramidalBMRegistration, "-r", os.path.join("Patients_Tensors", dwiPrefix + "_ADC.nrrd"),
                         "-m", T1Prefix + "_masked.nrrd", "-o", os.path.join(tmpFolder, "T1_reg_rig.nrrd"), "-O", os.path.join(tmpFolder, "T1_reg_rig_tr.txt"),
                         "-p", "4", "-l", "1", "--sp", "2", "-s", "0"]

if args.register_t1_on_dwi is True:
    t1RegistrationCommand += ["-I", "1"]
else:
    t1RegistrationCommand += ["-I", "0"]

call(t1RegistrationCommand)

command = [animaTransformSerieXmlGenerator, "-i", os.path.join(tmpFolder, "T1_reg_rig_tr.txt"), "-i", os.path.join(tmpFolder, "Patient_aff_tr.txt"),
           "-i", os.path.join(tmpFolder, dwiPrefix + "_nl_tr.nrrd"), "-o", os.path.join(tmpFolder, "Patient_T1_nl_tr.xml")]
call(command)

# Process tracks: augmenting with patient and perform comparison
for track in tracksLists:
    # augment tracks of the atlas with MCM patient data
    propsExtractionCommand = [animaTracksMCMPropertiesExtraction, "-i", os.path.join(args.raw_tracts_folder, track + '.fds'),
                              "-m", os.path.join('Transformed_Patients_MCM', dwiPrefix + "_MCM_avg_onAtlas.mcm"),
                              "-o", os.path.join('Patients_Augmented_Tracts', track + '_' + dwiPrefix + '_MCM_augmented_onAtlas.fds')]
    call(propsExtractionCommand)

    # Compare to controls list of augmented tracts
    propsComparisonCommand = [animaPatientToGroupComparisonOnTracks, "-i", os.path.join('Patients_Augmented_Tracts', track + '_' + dwiPrefix + '_MCM_augmented_onAtlas.fds'),
                              "-l", os.path.join(args.tracts_folder, "listData_" + track + ".txt"),
                              "-o", os.path.join(tmpFolder, track + '_' + dwiPrefix + '_PV.fds'),
                              "-a", os.path.join('Patients_Augmented_Tracts', track + '_' + dwiPrefix + '_controls_avg.fds')]
    call(propsComparisonCommand)

    fdrCorrectionCommand = [animaFibersFDRCorrectPValues, "-i", os.path.join(tmpFolder, track + '_' + dwiPrefix + '_PV.fds'),
                            "-o", os.path.join('Patients_Augmented_Tracts', track + '_' + dwiPrefix + '_FDR.fds'), "-q", "0.05"]
    call(fdrCorrectionCommand)

    # Bring back fibers into native image space
    bringFibersBackCommand = [animaFibersApplyTransformSerie, "-i", os.path.join('Patients_Augmented_Tracts', track + '_' + dwiPrefix + '_MCM_augmented_onAtlas.fds'), "-I",
                              "-t", os.path.join(tmpFolder, "Patient_T1_nl_tr.xml"), "-o", os.path.join('Patients_Augmented_Tracts', track + '_' + dwiPrefix + '_MCM_augmented.fds')]
    call(bringFibersBackCommand)

    bringFibersBackCommand = [animaFibersApplyTransformSerie, "-i", os.path.join('Patients_Augmented_Tracts', track + '_' + dwiPrefix + '_FDR.fds'), "-I",
                              "-t", os.path.join(tmpFolder, "Patient_T1_nl_tr.xml"), "-o", os.path.join('Patients_Augmented_Tracts', track + '_' + dwiPrefix + '_FDR.fds')]
    call(bringFibersBackCommand)

    bringFibersBackCommand = [animaFibersApplyTransformSerie, "-i", os.path.join('Patients_Augmented_Tracts', track + '_' + dwiPrefix + '_AC.fds'), "-I",
                              "-t", os.path.join(tmpFolder, "Patient_T1_nl_tr.xml"), "-o", os.path.join('Patients_Augmented_Tracts', track + '_' + dwiPrefix + '_AC.fds')]
    call(bringFibersBackCommand)

    bringFibersBackCommand = [animaFibersApplyTransformSerie, "-i", os.path.join('Patients_Augmented_Tracts', track + '_' + dwiPrefix + '_controls_avg.fds'), "-I",
                              "-t", os.path.join(tmpFolder, "Patient_T1_nl_tr.xml"), "-o", os.path.join('Patients_Augmented_Tracts', track + '_' + dwiPrefix + '_controls_avg.fds')]
    call(bringFibersBackCommand)

    # Compute final scores CSV
    command = [animaFibersDiseaseScores, "-i", os.path.join('Patients_Augmented_Tracts', track + '_' + dwiPrefix + '_FDR.fds'),
               "-o", os.path.join('Patients_Disease_Scores', track + '_' + dwiPrefix + '.csv'), "-p", "6"]
    call(command)

shutil.rmtree(tmpFolder)
