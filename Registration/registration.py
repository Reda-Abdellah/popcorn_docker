import os
import platform
import subprocess
import numpy as np
from subprocess import Popen, PIPE, STDOUT
import logging
from shutil import copyfile
import nibabel as nii

#refernce_MNI='final_mni_icbm152_t1_tal_nlin_sym_09c.nii'
#refernce_mask_MNI='final_mni_icbm152_t1_tal_nlin_sym_09c_mask.nii'
refernce_MNI='Registration/MNI_miplab-flair_sym_with-skull.nii'
refernce_mask_MNI='Registration/MNI_miplab-flair_sym_mask.nii'

def read_transform(filename):  # Simplified version of read_transform
    f = open(filename)
    text = f.read().split()
    M = np.eye(4)
    index = 9
    for i in range(4):
        if index >= 21:  # We read until word 21
            break
        for j in range(3):
            M[j, i] = np.float64(text[index])
            index += 1
    return M

def QC(fname, filename= refernce_MNI):
    mask = nii.load(
        filename=refernce_mask_MNI)
    mask = mask.get_fdata()
    T1 = nii.load(filename= filename)
    T1 = T1.get_fdata()
    ima = nii.load(filename=fname)
    ima = ima.get_fdata()

    ind = np.where(np.isnan(ima))
    cc = np.corrcoef(T1[ind] * mask[ind],
                     ima[ind] * mask[ind])
    correlation1 = cc[0][1]
    cc = np.corrcoef(T1[ind], ima[ind])
    correlation2 = cc[0][1]
    return correlation1, correlation2

def run_cmd(cmd):
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    output = p.stdout.read()
    print(output)

def MASK_to_MNI(affine_intoMNI, in_flair, registred_flair):
    bin_name = 'Registration/antsApplyTransforms'
    #interpolation='GenericLabel[Linear]'
    interpolation='MultiLabel[0.3,0]'
    cmd= bin_name+' -d 3 -i '+in_flair+' -r '+refernce_MNI+' -o '+registred_flair+' -n '+interpolation+' -t '+affine_intoMNI
    run_cmd(cmd)
    return True

def FLAIR_to_MNI(affine_intoMNI, in_flair, registred_flair):
    bin_name = 'Registration/antsApplyTransforms'
    cmd= bin_name+' -d 3 -i '+in_flair+' -r '+refernce_MNI+' -o '+registred_flair+' -n BSpline -t '+affine_intoMNI
    run_cmd(cmd)
    return True

def ToMNI_ANTS_ref(filename):
    # Parsing filenames
    # Getting image namefile  ###########
    ima1_name = filename.rsplit('/', 1)[-1]
    ft = filename.replace(ima1_name, 'affine_'+ima1_name)
    ft2 = ft.replace('.nii', 'Affine.txt')
    bin_name1 = 'Registration/ANTS'  # Declaring both binary name files for os compatible
    args = (bin_name1, '3', '-m', 'MI['+refernce_MNI+',' +
            filename+',1,32]', '-i', '0', '-o', ft)
    subprocess.run(args)
    args = (bin_name1, '3', '-m', 'MI['+refernce_MNI+','+filename+',1,32]', '-i', '0' '-o',
            ft, '--mask-image', refernce_mask_MNI, '--initial-affine', ft2)
    subprocess.run(args)
    return  ft2

def to_native(outputname,inputname,transform_name,reference_name):
    ants_bin='Registration/antsApplyTransforms'
    #ants_bin='../Compilation_lesionBrain_v10/Registration/antsApplyTransforms'
    command=ants_bin+' -d 3 -i ' + inputname+' -r ' + reference_name +' -o ' +outputname + ' -n MultiLabel[0.3,0] -t [' + transform_name +', 1]';
    print(str(command))
    os.system(str(command))
    return outputname
