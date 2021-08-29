from sklearn.datasets import load_digits
import os, random
import glob
import numpy as np
import nibabel as nii
import math
import operator
from scipy.ndimage.interpolation import zoom
from keras.models import load_model
from scipy import ndimage
import scipy.io as sio
import modelos
import statsmodels.api as sm
from scipy.signal import argrelextrema
from collections import OrderedDict, defaultdict
from skimage import measure
from scipy.stats import pearsonr
from keras import backend as K
import time
import tensorflow as tf

def normalize_image(vol, contrast):
	# copied from FLEXCONN
	# slightly changed to fit our implementation
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

def copyModel2Model(model_source, model_target, certain_layer=""):
	#print(len(model_target.layers))
	cpt =0
	for l_tg,l_sr in zip(model_target.layers, model_source.layers):
		wk0=l_sr.get_weights()
		l_tg.set_weights(wk0)
		print("copy layer " + str(cpt) + " :" + l_tg.name)
		if l_sr.name == certain_layer:
			l_tg.name=l_sr.name
		#if cpt == len(model_target.layers)-2:
			print("model source was copied into model target")
			return True
		cpt = cpt+1
	print("model source was copied into model target")

def load_seg(path):
	seg_img = nii.load(path)
	seg=seg_img.get_data()
	seg=seg.astype('int')
	return seg

def keyword_toList(path,keyword):
	search=os.path.join(path,'*'+keyword+'*')
	lista=sorted(glob.glob(search))
	print("list contains: "+str( len(lista))+" elements")
	return lista

def load_modalities(T1_name,FLAIR_name,MASK_name=None):
	T1_img = nii.load(T1_name)
	T1=T1_img.get_data()
	T1=T1.astype('float32')
	FLAIR_img = nii.load(FLAIR_name)
	FLAIR=FLAIR_img.get_data()
	FLAIR=FLAIR.astype('float32')
	if(not MASK_name==None):
		MASK_img = nii.load(MASK_name)
		MASK = MASK_img.get_data()
		MASK=MASK.astype('int')
		T1=T1*MASK
		FLAIR=FLAIR*MASK
	peak = normalize_image(T1, 't1')
	T1=T1/peak
	peak = normalize_image(FLAIR, 'flair')
	FLAIR=FLAIR/peak
	return T1,FLAIR

def patches(T1,FLAIR,nbNN=[3,3,3]):
	crop_bg = 4
	ps1,ps2,ps3=96,96,96

	overlap1 = np.floor((nbNN[0]*ps1 - (T1.shape[0]-2*crop_bg)) / (nbNN[0]-1))
	offset1 = ps1 - overlap1.astype('int')
	overlap2 = np.floor((nbNN[1]*ps2 - (T1.shape[1]-2*crop_bg)) / (nbNN[1]-1))
	offset2 = ps2 - overlap2.astype('int')
	overlap3 = np.floor((nbNN[2]*ps3 - (T1.shape[2]-2*crop_bg)) / (nbNN[2]-1))
	offset3 = ps3 - overlap3.astype('int')
	pT1=patch_extract_3D_v2(T1,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
	pFLAIR=patch_extract_3D_v2(FLAIR,(ps1,ps2,ps3),nbNN,offset1,offset2,offset3,crop_bg)
	pT1 = np.expand_dims(pT1.astype('float32'),axis=4)
	pFLAIR =np.expand_dims(pFLAIR.astype('float32'),axis=4)
	x_in=np.concatenate((pT1,pFLAIR),axis=4)
	return x_in

def patch_extract_3D_v2(input,patch_shape,nbNN,offx=1,offy=1,offz=1,crop_bg=0):
	n=0
	numPatches=nbNN[0]*nbNN[1]*nbNN[2]
	local_patch = np.zeros((patch_shape[0],patch_shape[1],patch_shape[2]),input.dtype)
	patches_3D=np.zeros((numPatches,patch_shape[0],patch_shape[1],patch_shape[2]),input.dtype)
	for x in range(crop_bg,(nbNN[0]-1)*offx+crop_bg+1,offx):
		for y in range(crop_bg,(nbNN[1]-1)*offy+crop_bg+1,offy):
			for z in range(0,(nbNN[2]-1)*offz+1,offz): #spine is touching one side in Z dicrection, so crop has to be asymetric
				xx = x+patch_shape[0]
				if xx> input.shape[0]:
					xx = input.shape[0]
				yy = y+patch_shape[1]
				if yy> input.shape[1]:
					yy = input.shape[1]
				zz = z+patch_shape[2]
				if zz> input.shape[2]:
					zz = input.shape[2]
				# To deal with defferent patch size due to border issue
				local_patch = local_patch*0
				local_patch[0:xx-x,0:yy-y,0:zz-z] = input[x:xx,y:yy,z:zz]
				a=np.reshape(local_patch,(1,patches_3D.shape[1],patches_3D.shape[2],patches_3D.shape[3]))
				patches_3D[n,:,:,:]=a
				n=n+1
	patches_3D=patches_3D[0:n,:,:,:]
	return patches_3D

def patch_reconstruct_3D_v2(out_shape,patches,nbNN, offx=1,offy=1,offz=1,crop_bg=0):
	n=0
	output=np.zeros(out_shape,patches.dtype)
	acu=np.zeros(out_shape,patches.dtype)
	pesos=np.ones((patches.shape[1],patches.shape[2],patches.shape[3]))
	for x in range(crop_background_border,(nbNN-1)*offx+crop_background_border+1,offx):
		for y in range(crop_background_border,(nbNN-1)*offy+crop_background_border+1,offy):
			for z in range(crop_background_border,(nbNN-1)*offz+crop_background_border+1,offz):

				xx = x+patches.shape[1]
				if xx> input.shape[0]:
					xx = input.shape[0]
				yy = y+patches.shape[2]
				if yy> input.shape[1]:
					yy = input.shape[1]
				zz = z+patches.shape[3]
				if zz> input.shape[2]:
					zz = input.shape[2]

				output[x:xx,y:yy,z:zz]=output[x:xx,y:yy,z:zz]+np.reshape(patches[n,:,:,:],(patches.shape[1],patches.shape[2],patches.shape[3]))
				acu[x:xx,y:yy,z:zz]=acu[x:xx,y:yy,z:zz]+pesos
				n=n+1
	ind=np.where(acu==0)
	acu[ind]=1
	output=output/acu
	return output

def seg_majvote(T1,FLAIR,model,nbNN=[5,5,5],ps=[96,96,96],multi_out=False):
	MASK = (1-(T1==0).astype('int'))
	ind=np.where(MASK>0)
	indbg=np.where(MASK==0)
	crop_bg = 4
	overlap1 = np.floor((nbNN[0]*ps[0] - (T1.shape[0]-2*crop_bg)) / (nbNN[0]-1))
	offset1 = ps[0] - overlap1.astype('int')
	overlap2 = np.floor((nbNN[1]*ps[1] - (T1.shape[1]-2*crop_bg)) / (nbNN[1]-1))
	offset2 = ps[1] - overlap2.astype('int')
	overlap3 = np.floor((nbNN[2]*ps[2] - (T1.shape[2]-2*crop_bg)) / (nbNN[2]-1))
	offset3 = ps[2]- overlap3.astype('int')
	pT1=patch_extract_3D_v2(T1,(ps[0],ps[1],ps[2]),nbNN,offset1,offset2,offset3,crop_bg)
	pT1= pT1.astype('float32')
	pFLAIR=patch_extract_3D_v2(FLAIR,(ps[0],ps[1],ps[2]),nbNN,offset1,offset2,offset3,crop_bg)
	pFLAIR= pFLAIR.astype('float32')

	out_shape=(T1.shape[0],T1.shape[1],T1.shape[2],2)
	output=np.zeros(out_shape,T1.dtype)
	acu=np.zeros(out_shape[0:3],T1.dtype)

	ii=0 # Network ID

	for x in range(crop_bg,(nbNN[0]-1)*offset1+crop_bg+1,offset1):
		for y in range(crop_bg,(nbNN[1]-1)*offset2+crop_bg+1,offset2):
			for z in range(0,(nbNN[2]-1)*offset3+1,offset3):


				T = np.reshape(pT1[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
				F = np.reshape(pFLAIR[ii], (1,pFLAIR.shape[1],pFLAIR.shape[2],pFLAIR.shape[3], 1))
				T=np.concatenate((T,F), axis=4)


				if(not ii< int(((nbNN[0]+1)/2)*nbNN[1]*nbNN[2])):

					T=T[:,-1::-1,:,:,:]


				lista=np.array([0,1])
				if(multi_out):
					patches = model.predict(T)[0]
				else:
					patches = model.predict(T)

				if(not ii< int(((nbNN[0]+1)/2)*nbNN[1]*nbNN[2])):
					patches=patches[:,-1::-1,:,:,:]


				xx = x+patches.shape[1]
				if xx> output.shape[0]:
					xx = output.shape[0]

				yy = y+patches.shape[2]
				if yy> output.shape[1]:
					yy = output.shape[1]

				zz = z+patches.shape[3]
				if zz> output.shape[2]:
					zz = output.shape[2]

				#store result
				local_patch = np.reshape(patches,(patches.shape[1],patches.shape[2],patches.shape[3],patches.shape[4]))

				output[x:xx,y:yy,z:zz,:]=output[x:xx,y:yy,z:zz,:]+local_patch[0:xx-x,0:yy-y,0:zz-z]
				acu[x:xx,y:yy,z:zz]=acu[x:xx,y:yy,z:zz]+1#pesos

				ii=ii+1

	ind=np.where(acu==0)
	mask_ind = np.where(acu>0)
	acu[ind]=1

	SEG= np.argmax(output, axis=3)
	SEG= np.reshape(SEG, SEG.shape[0:3])
	SEG_mask = SEG*MASK


	return SEG_mask
