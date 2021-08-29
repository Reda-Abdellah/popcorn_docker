from prediction import *
from preprocessing import *
import json,os,shutil
from report_utils import *
from make_reports import *
from utils import *
import argparse


parser = argparse.ArgumentParser(
    description="""Blabla""", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-f', '--flair_keyword', type=str, required=True)
#parser.add_argument('-t', '--t1_keyword', type=str, required=True)
parser.add_argument('-i', '--in_folder_path', type=str, required=True)
parser.add_argument('--no_preprocessing', action='store_true')
args = parser.parse_args()
in_folder_path=args.in_folder_path

Weights_list= keyword_toList(path='/Weights/',keyword='.h5')

#preprocessing for training
if(args.no_preprocessing):
    T1_list= keyword_toList(path=in_folder_path,keyword=args.t1_keyword)
    FLAIR_list= keyword_toList(path=in_folder_path,keyword=args.FLAIR_keyword)
    segmentation(nbNN=[5,5,5],ps=[96,96,96],
            Weights_list=Weights_list,
            T1_list=T1_list, FLAIR_list=FLAIR_list,
            FG_list=None, normalization="kde")

else:
    mni_T1s,mni_FLAIRs,mni_MASKSs,intoT1s,affines,listaT1,listaFLAIR= preprocess_folder(
                in_folder_path,
                T1Keyword=args.t1_keyword,
                FLAIRKeyword=args.flair_keyword)

    lesions_list= segmentation(nbNN=[5,5,5],ps=[96,96,96],
            Weights_list=Weights_list,
            T1_list=mni_T1s, FLAIR_list=mni_FLAIRs,
            FG_list=mni_MASKSs, normalization="kde")

    for mni_lesions_name, mni_mask_name, nativeT1_name, to_mni_affine in zip(lesions_list, mni_MASKSs, listaT1,affines):
        to_native(mni_lesions_name,to_mni_affine,nativeT1_name)
        to_native(mni_mask_name,to_mni_affine,nativeT1_name)

for img in keyword_toList(path=in_folder_path,keyword='.nii'):
    if( not ('.gz' in img) ):
        os.system('gzip '+img)
