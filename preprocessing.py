import glob,os
from shutil import copyfile
from utils import *

def preprocess_folder(in_folder_path,T1Keyword="T1",FLAIRKeyword='FLAIR'): #receives absolute path
    listaT1 = keyword_toList(in_folder_path,T1Keyword)
    listaFLAIR = keyword_toList(in_folder_path,FLAIRKeyword)
    mni_FLAIRs=[]
    mni_T1s=[]
    mni_MASKSs=[]
    intoT1s=[]
    affines=[]
    for T1,FLAIR in zip(listaT1,listaFLAIR):
        print('processing: '+T1+' and '+FLAIR)
        nativeT1_name=in_folder_path+'native_'+T1.split('/')[-1]
        nativeFLAIR_name= in_folder_path+'native_'+FLAIR.split('/')[-1]
        nativeT1_name=nativeT1_name.replace('.gz','')
        nativeFLAIR_name=nativeFLAIR_name.replace('.gz','')
        if('.gz' in FLAIR):
            copyfile(FLAIR, nativeFLAIR_name+'.gz')
            os.system('gunzip '+nativeFLAIR_name+'.gz')
        else:
            copyfile(FLAIR, nativeFLAIR_name)
        if('.gz' in T1):
            copyfile(T1, nativeT1_name+'.gz')
            os.system('gunzip '+nativeT1_name+'.gz')
        else:
            copyfile(T1, nativeT1_name)
        newT1, newFLAIR, new_mask, new_intot1, new_affine= preprocess_file(nativeT1_name, nativeFLAIR_name)
        mni_T1s.append(newT1)
        mni_FLAIRs.append(newFLAIR)
        mni_MASKSs.append(new_mask)
        intoT1s.append(new_intot1)
        affines.append(new_affine)
    return mni_T1s,mni_FLAIRs,mni_MASKSs,intoT1s,affines,listaT1,listaFLAIR


def preprocess_file(nativeT1_name, nativeFLAIR_name): #receives absolute path
    print('processing: '+nativeT1_name+' and '+nativeFLAIR_name)
    newT1=nativeT1_name.replace('native_','preprocessed_mni_')
    newFLAIR=nativeFLAIR_name.replace('native_','preprocessed_mni_')
    copyfile(nativeT1_name,newT1)
    copyfile(nativeFLAIR_name,newFLAIR)
    bin='./lesionBrain_v11_fullpreprocessing_exe'
    command=bin+' '+nativeT1_name+' '+nativeFLAIR_name
    os.system(command)
    outT1=nativeT1_name.replace('.nii','_check.nii')
    outT1=outT1.replace('native_','n_mfmni_fnative_')
    out_intot1=nativeFLAIR_name.replace('.nii','_checkAffine.txt')
    out_intot1=out_intot1.replace('native_','affine_intot1_fnative_')
    out_affine=nativeT1_name.replace('.nii','_checkAffine.txt')
    out_affine=out_affine.replace('native_','affine_mfnative_')
    outFLAIR=outT1.replace('t1', 'flair')
    outMASK=outT1.replace('n_mfmni_','mask_n_mfmni_')
    out_crisp_filename=outT1.replace('n_mfmni_','crisp_mfmni_')
    out_hemi_fileneame= outT1.replace('n_mfmni_','hemi_n_mfmni_')
    out_structures_filename = outT1.replace('n_mfmni_','lab_n_mfmni_')
    new_crisp_filename=newT1.replace('preprocessed','crisp').replace('_t1','')
    new_hemi_fileneame= newT1.replace('preprocessed','hemi').replace('_t1','')
    new_structures_filename = newT1.replace('preprocessed','structure').replace('_t1','')
    new_mask=newT1.replace('t1','mask')
    new_intot1=nativeFLAIR_name.replace('native_','affine_intot1_fmni_').replace('.nii','.txt')
    new_affine=nativeT1_name.replace('native_','affine_mfmni_').replace('.nii','.txt')
    os.rename(outT1, newT1)
    os.rename(out_hemi_fileneame, new_hemi_fileneame)
    os.rename(out_crisp_filename, new_crisp_filename)
    os.rename(out_structures_filename, new_structures_filename)
    os.rename(outFLAIR, newFLAIR)
    os.rename(outMASK, new_mask)
    os.rename(out_intot1, new_intot1)
    os.rename(out_affine, new_affine)
    os.remove(nativeT1_name.replace('.nii','_check.nii'))
    os.remove(nativeFLAIR_name.replace('.nii','_check.nii'))
    return newT1, newFLAIR, new_mask, new_intot1, new_affine, new_crisp_filename, new_hemi_fileneame, new_structures_filename


def ground_truth_toMNI(in_folder_path,preprocessed_out_folder,SEG_keyword):
    ants_bin='./Registration/antsApplyTransforms'
    for seg_keyword in SEG_keyword:
        listaSEG = keyword_toList(in_folder_path,seg_keyword)
        for inputname in listaSEG:
            if('.gz' in inputname):
                copyfile(inputname,'tmp.nii.gz')
                os.system('gunzip '+'tmp.nii.gz')
                outputname= inputname.replace(seg_keyword,seg_keyword+'_MNI_')
                outputname=outputname.replace('.gz','')
                command=ants_bin+' -d 3 tmp.nii -r ' + reference_name +' -o ' +outputname + ' -n MultiLabel[0.3,0] -t [' + transform_name +', 1]';
                os.system(command)
                os.remove('tmp.nii')
            else:
                outputname= inputname.replace(seg_keyword,seg_keyword+'_MNI_')
                command=ants_bin+' -d 3 ' + inputname+' -r ' + reference_name +' -o ' +outputname + ' -n MultiLabel[0.3,0] -t [' + transform_name +', 1]';
                os.system(command)
    files_list=keyword_toList(preprocessed_out_folder,'.')
    for file in files_list:
        if(not ('.gz' in file) ):
            os.system('gzip '+file)
