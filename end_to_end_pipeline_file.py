from prediction import *
from preprocessing import *
import json,os,shutil
from report_utils import *
from make_reports import *
from utils import *
import argparse


parser = argparse.ArgumentParser(
    description="""Blabla""", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-f', '--flair', type=str, required=True)
#parser.add_argument('-t', '--t1', type=str, required=True)
parser.add_argument('-o', '--index_name', type=str, required=True)
parser.add_argument('--no_report', action='store_true')
parser.add_argument( '--sex', type=str, default='Uknown')
parser.add_argument('--age', type=str, default='Uknown')
args = parser.parse_args()

nativeFLAIR_name='/tmp/native_'+args.index_name+'_flair.nii'
nativeT1_name='/tmp/native_'+args.index_name+'_t1.nii'

####get generated T1####
preprocessed_flair_name= '/data/patients_preprocessed/patient_X/flair.nii'
registred_flair_name= '/data/patients_preprocessed/patient_X/preprocessed_mni_flair.nii'
brain_mask_name= '/data/patients_preprocessed/patient_X/brain_mask.nii'
flair_name= '/data/patients/patient_X/flair.nii'

if('.gz' in args.flair):
    shutil.copyfile(args.flair, flair_name+'.gz')
    os.system('gunzip '+flair_name+'.gz')
else:
    shutil.copyfile(args.flair, flair_name)
cmd= "python /anima/Anima-Scripts-Public/ms_lesion_segmentation/animaMSLongitudinalPreprocessing.py -i /data/patients/ -o /data/patients_preprocessed/"
os.system(cmd)

affine_flair_intoMNI = ToMNI_ANTS_ref( preprocessed_flair_name)
FLAIR_to_MNI(affine_flair_intoMNI, preprocessed_flair_name, registred_flair_name)
MASK_to_MNI(affine_flair_intoMNI, brain_mask_name, brain_mask_name)
preprocessed_flair_name=registred_flair_name
t1_name= registred_flair_name.replace('flair', 'generated_t1')
t1_generated_native=args.flair.replace('.nii','_native_gen_t1.nii')
get_syntetic_t1(registred_flair_name, brain_mask_name, t1_name)
to_native_t1(t1_name,t1_generated_native,affine_flair_intoMNI,flair_name)

########################

if('.gz' in args.flair):
    shutil.copyfile(args.flair, nativeFLAIR_name+'.gz')
    os.system('gunzip '+nativeFLAIR_name+'.gz')
else:
    shutil.copyfile(args.flair, nativeFLAIR_name)

if('.gz' in t1_generated_native):
    shutil.copyfile(t1_generated_native, nativeT1_name+'.gz')
    os.system('gunzip '+nativeT1_name+'.gz')
else:
    shutil.copyfile(t1_generated_native, nativeT1_name)



mni_T1_name, mni_flair_name, mni_mask_name, intot1, to_mni_affine, crisp_filename, hemi_fileneame, structures_filename = preprocess_file(nativeT1_name, nativeFLAIR_name)
Weights_list= keyword_toList(path='/Weights/',keyword='.h5')

get_lesions(mni_lesions_name,mni_flair_name,mni_mask_name)

native_lesion= to_native(mni_lesions_name,to_mni_affine,nativeT1_name)
native_mask= to_native(mni_mask_name,to_mni_affine,nativeT1_name)
unfiltred_t1_filename= mni_T1_name.replace('t1', 'unfiltred')
to_MNI(nativeT1_name,unfiltred_t1_filename,nativeT1_name,mni_T1_name)

os.system('gzip '+mni_mask_name)
os.system('gzip '+mni_flair_name)
os.system('gzip '+mni_T1_name)
os.system('gzip '+mni_lesions_name)
os.system('gzip '+nativeFLAIR_name)
os.system('gzip '+nativeT1_name)
os.system('gzip '+native_lesion)
os.system('gzip '+native_mask)
os.system('gzip '+crisp_filename)
os.system('gzip '+hemi_fileneame)
os.system('gzip '+structures_filename)


if(not args.no_report):
    report(unfiltred_t1_filename, mni_T1_name+'.gz', mni_flair_name+'.gz', mni_mask_name+'.gz', mni_lesions_name+'.gz',
            to_mni_affine,crisp_filename+'.gz', hemi_fileneame+'.gz',structures_filename+'.gz', args.age, args.sex)
os.remove(unfiltred_t1_filename)
