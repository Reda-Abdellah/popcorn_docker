import  os, glob,  gc
import nibabel as nii
import numpy as np
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import statsmodels.api as sm
from scipy.signal import argrelextrema
from keras import optimizers
import modelos
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seg_majvote_keras(FLAIR,model,WEIGHTS,ps=[64,64,64],
        offset1=32,offset2=32,offset3=32,crop_bg=0):
    MASK = (1-(FLAIR==0).astype('int'))
    ind=np.where(MASK>0)
    indbg=np.where(MASK==0)

    out_shape=(FLAIR.shape[0],FLAIR.shape[1],FLAIR.shape[2],2)
    output=np.zeros(out_shape,FLAIR.dtype)
    acu=np.zeros(out_shape[0:3],FLAIR.dtype)


    ii=0 # Network ID
    for model_name in WEIGHTS:
        model.load_weights(model_name)
        model.compile(optimizer=optimizers.Adam(0.0001), loss='mse')
        for x in range(crop_bg,  FLAIR.shape[0] ,offset1):
            xx = x+ps[0]
            if xx> output.shape[0]:
                xx = output.shape[0]
                x=xx-ps[0]
            for y in range(crop_bg,  FLAIR.shape[1] ,offset2):
                yy = y+ps[1]
                if yy> output.shape[1]:
                    yy = output.shape[1]
                    y=yy-ps[1]
                for z in range(crop_bg,  FLAIR.shape[2] ,offset3):
                    zz = z+ps[2]
                    if zz> output.shape[2]:
                        zz = output.shape[2]
                        z=zz-ps[2]

                    T = np.reshape(   FLAIR[x:xx,y:yy,z:zz] , (1,ps[0],ps[1],ps[2], 1))
                    patches=model.predict(T)
                    #store result
                    local_patch = np.reshape(patches,(patches.shape[1],patches.shape[2],patches.shape[3],patches.shape[4]))
                    output[x:xx,y:yy,z:zz,:]=output[x:xx,y:yy,z:zz,:]+local_patch[0:xx-x,0:yy-y,0:zz-z]
                    ii=ii+1

    SEG= np.argmax(output, axis=3)
    SEG_mask= np.reshape(SEG, SEG.shape[0:3])
    return SEG_mask

def seg_majvote(FLAIR,WEIGHTS,ps=[64,64,64],
        offset1=32,offset2=32,offset3=32,crop_bg=0):
    MASK = (1-(FLAIR==0).astype('int'))
    ind=np.where(MASK>0)
    indbg=np.where(MASK==0)

    out_shape=(FLAIR.shape[0],FLAIR.shape[1],FLAIR.shape[2],2)
    output=np.zeros(out_shape,FLAIR.dtype)
    acu=np.zeros(out_shape[0:3],FLAIR.dtype)


    ii=0 # Network ID
    for model_name in WEIGHTS:
        model= torch.load(model_name)
        for x in range(crop_bg,  FLAIR.shape[0] ,offset1):
            xx = x+ps[0]
            if xx> output.shape[0]:
                xx = output.shape[0]
                x=xx-ps[0]
            for y in range(crop_bg,  FLAIR.shape[1] ,offset2):
                yy = y+ps[1]
                if yy> output.shape[1]:
                    yy = output.shape[1]
                    y=yy-ps[1]
                for z in range(crop_bg,  FLAIR.shape[2] ,offset3):
                    zz = z+ps[2]
                    if zz> output.shape[2]:
                        zz = output.shape[2]
                        z=zz-ps[2]

                    T = np.reshape(   FLAIR[x:xx,y:yy,z:zz] , (1,ps[0],ps[1],ps[2], 1))
                    T=torch.from_numpy(T.transpose((0,4,1,2,3))).to(device)
                    patches = model(T)
                    patches= patches.cpu().numpy()
                    patches= patches.transpose((0,2,3,4,1))
                    #store result
                    local_patch = np.reshape(patches,(patches.shape[1],patches.shape[2],patches.shape[3],patches.shape[4]))
                    output[x:xx,y:yy,z:zz,:]=output[x:xx,y:yy,z:zz,:]+local_patch[0:xx-x,0:yy-y,0:zz-z]
                    ii=ii+1

    SEG= np.argmax(output, axis=3)
    SEG_mask= np.reshape(SEG, SEG.shape[0:3])
    return SEG_mask

def load_time(FLAIR_1_name,MASK_name=None):
    FLAIR_img = nii.load(FLAIR_1_name)
    FLAIR_1=FLAIR_img.get_data()
    FLAIR_1=FLAIR_1.astype('float32')
    if(not MASK_name==None):
        MASK_img = nii.load(MASK_name)
        MASK = MASK_img.get_data()
        MASK=MASK.astype('int')
        FLAIR_1=FLAIR_1*MASK
    peak = normalize_image(FLAIR_1, 'flair')
    FLAIR_1=FLAIR_1/peak
    return FLAIR_1

def seg_region(flair, overlap=32):
    ax0= flair.sum(axis=(1,2))
    ax0=np.where(ax0>0)
    #ax0min=ax0[0][0]
    #ax0max=ax0[0][-1]
    ax0min=min(ax0[0])
    ax0max=max(ax0[0])
    ax1= flair.sum(axis=(0,2))
    ax1=np.where(ax1>0)
    #ax1min=ax1[0][0]
    #ax1max=ax1[0][-1]
    ax1min=min(ax1[0])
    ax1max=max(ax1[0])
    ax2= flair.sum(axis=(0,1))
    ax2=np.where(ax2>0)
    #ax2min=ax2[0][0]
    #ax2max=ax2[0][-1]
    ax2min=min(ax2[0])
    ax2max=max(ax2[0])

    if(overlap>0):
        ax0min=max([ax0min-overlap,0])
        ax0max=min([ax0max+overlap, flair.shape[0] ])
        ax1min=max([ax1min-overlap,0])
        ax1max=min([ax1max+overlap, flair.shape[1] ])
        ax2min=max([ax2min-overlap,0])
        ax2max=min([ax2max+overlap, flair.shape[2] ])

    return ax0min,ax0max,ax1min,ax1max,ax2min,ax2max

def normalize_image(vol, contrast):
    # copied  MedICL-VU / LesionSeg
    temp = vol[np.nonzero(vol)].astype(float)
    q = np.percentile(temp, 99)
    temp = temp[temp <= q]
    temp = temp.reshape(-1, 1)
    bw = q / 80
    # print("99th quantile is %.4f, gridsize = %.4f" % (q, bw))

    kde = sm.nonparametric.KDEUnivariate(temp)

    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    x_mat = 100.0 * kde.density
    y_mat = kde.support

    indx = argrelextrema(x_mat, np.greater)
    indx = np.asarray(indx, dtype=int)
    heights = x_mat[indx][0]
    peaks = y_mat[indx][0]
    peak = 0.00

    if contrast.lower() in ["t1", "mprage"]:
        peak = peaks[-1]
    elif contrast.lower() in ['t2', 'pd', 'flair', 'fl']:
        peak_height = np.amax(heights)
        idx = np.where(heights == peak_height)
        peak = peaks[idx]
    else:
        print("Contrast must be either t1,t2,pd, or flair. You entered %s. Returning 0." % contrast)

    return peak

def get_lesions_keras(pred_name,flair1_name,brain_mask_name):
    WEIGHTS= sorted(glob.glob("/anima/WEIGHTS/*"))
    FLAIR_1 =load_time(flair1_name, brain_mask_name)
    model=modelos.load_UNET3D_SLANT27_v2_groupNorm(64,64,64,1,2,24,0)

    SEG_mask=seg_majvote(FLAIR_1,model,WEIGHTS,ps=[64,64,64],
                        offset1=32,offset2=32,offset3=32,crop_bg=0)
    img = nii.Nifti1Image(SEG_mask.astype(np.uint8), nii.load(flair1_name).affine )
    img.to_filename(pred_name)
    gc.collect() #free memory

def get_lesions(pred_name,flair1_name,brain_mask_name):
    WEIGHTS= sorted(glob.glob("/anima/WEIGHTS/*"))
    #WEIGHTS= sorted(glob.glob("/mnt/4TB/POPCORN_docker/for_train/Weights/POPCORN_cocktail/*"))
    FLAIR_1 =load_time(flair1_name, brain_mask_name)

    SEG_mask=seg_majvote(FLAIR_1,WEIGHTS,ps=[64,64,64],
                        offset1=32,offset2=32,offset3=32,crop_bg=0)
    img = nii.Nifti1Image(SEG_mask.astype(np.uint8), nii.load(flair1_name).affine )
    img.to_filename(pred_name)
    gc.collect() #free memory
#get_lesions('seg.nii.gz','yamato_training_01_01_flair.nii.gz',brain_mask_name=None)
