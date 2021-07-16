import glob, os, shutil
from test_utils import *
import configparser as ConfParser
import argparse
from Registration.registration import ToMNI_ANTS_ref, FLAIR_to_MNI, MASK_to_MNI, to_native
#from report_utils import *
#from make_reports import *

register=True
# Argument parsing
parser = argparse.ArgumentParser(
    description="""Blabla""", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-i', '--flair', type=str, required=True)
parser.add_argument('-o', '--index_name', type=str, required=True)
parser.add_argument('--no_registration', action='store_true')
#parser.add_argument('--report', action='store_true')


args = parser.parse_args()
preprocessed_flair_name= '/data/patients_preprocessed/patient_X/flair.nii'
registred_flair_name= '/data/patients_preprocessed/patient_X/preprocessed_mni_flair.nii'
brain_mask_name= '/data/patients_preprocessed/patient_X/brain_mask.nii'
flair_name= '/data/patients/patient_X/flair.nii'

out_seg_name= '/tmp/native_'+args.index_name+'_seg.nii'
out_seg_mni_name= '/tmp/mni_'+args.index_name+'_seg.nii'
out_preprocessed_flair='/tmp/preprocessed_'+args.index_name+'_flair.nii'
out_preprocessed_flair_mni='/tmp/preprocessed_mni_'+args.index_name+'_flair.nii'
out_brain_mask='/tmp/'+args.index_name+'_mask.nii'
out_brain_mask_mni='/tmp/mni_'+args.index_name+'_mask.nii'

do_register= not (args.no_registration)

if('.gz' in args.flair):
    shutil.copyfile(args.flair, flair_name+'.gz')
    os.system('gunzip '+flair_name+'.gz')
else:
    shutil.copyfile(args.flair, flair_name)

cmd= "python /anima/Anima-Scripts-Public/ms_lesion_segmentation/animaMSLongitudinalPreprocessing.py -i /data/patients/ -o /data/patients_preprocessed/"
os.system(cmd)

if(do_register):
    shutil.copyfile(preprocessed_flair_name, out_preprocessed_flair)
    shutil.copyfile(brain_mask_name, out_brain_mask)

"""
if('.gz' in preprocessed_flair_name):
        os.system('gunzip '+preprocessed_flair_name)
        preprocessed_flair_name=preprocessed_flair_name.replace('.gz','')

if('.gz' in brain_mask_name):
        os.system('gunzip '+brain_mask_name)
        brain_mask_name=brain_mask_name.replace('.gz','')
"""

if(do_register):
    affine_flair_intoMNI = ToMNI_ANTS_ref( preprocessed_flair_name)
    FLAIR_to_MNI(affine_flair_intoMNI, preprocessed_flair_name, registred_flair_name)
    MASK_to_MNI(affine_flair_intoMNI, brain_mask_name, brain_mask_name)
    preprocessed_flair_name=registred_flair_name


if(do_register):
    shutil.copyfile(preprocessed_flair_name, out_preprocessed_flair_mni)
    shutil.copyfile(brain_mask_name, out_brain_mask_mni )
else:
    shutil.copyfile(preprocessed_flair_name, out_preprocessed_flair)
    shutil.copyfile(brain_mask_name, out_brain_mask )
    #out_seg_mni_name=out_seg_name

if(do_register):
    get_lesions(out_seg_mni_name,preprocessed_flair_name,brain_mask_name)
else:
    get_lesions(out_seg_name,preprocessed_flair_name,brain_mask_name)

if(do_register):
    to_native(out_seg_name,out_seg_mni_name,affine_flair_intoMNI,out_preprocessed_flair)
    if(not ('.gz' in out_seg_mni_name)):
        os.system('gzip '+out_seg_mni_name)
    if(not ('.gz' in out_brain_mask_mni)):
        os.system('gzip '+out_brain_mask_mni)
    if(not ('.gz' in out_preprocessed_flair_mni)):
        os.system('gzip '+out_preprocessed_flair_mni)

if(not ('.gz' in out_brain_mask)):
    os.system('gzip '+out_brain_mask)
if(not ('.gz' in out_seg_name)):
    os.system('gzip '+out_seg_name)
if(not ('.gz' in out_preprocessed_flair)):
    os.system('gzip '+out_preprocessed_flair)

"""
if(do_register and args.report):
    report(out_preprocessed_flair, out_preprocessed_flair_mni, out_brain_mask_mni, out_seg_mni_name, affine_flair_intoMNI)
"""
