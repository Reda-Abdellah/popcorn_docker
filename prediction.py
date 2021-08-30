from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import numpy as np
from scipy import ndimage
import nibabel as nii
import os, glob, time, gc, datetime
import modelos
import scipy.misc
from scipy import signal
import multiprocessing as mp
from sklearn.metrics import confusion_matrix
from shutil import copyfile
import statsmodels.api as sm
from scipy.signal import argrelextrema
from collections import OrderedDict, defaultdict
from skimage import measure
from scipy.stats import pearsonr
from utils import *
from test_utils import *

def TTDO(X1,model):
	#test time dropout
	f = K.function([model.layers[0].input, K.learning_phase()],[model.layers[-1].output])
	result = np.asarray(f([X1,1])[0])
	return result

def get_syntetic_t1(flair_name, mask_name, t1_name):
	#load model
	nf=40
	FLAIR=load_time(flair1_name, brain_mask_name)
	FLAIR=np.pad(FLAIR,((0,1),(0,1),(0,1)),mode='symmetric')
	FLAIR = Subimage_extraction(FLAIR)
	ps1=np.floor(FLAIR.shape[1])
	ps2=np.floor(FLAIR.shape[2])
	ps3=np.floor(FLAIR.shape[3])
	ch =FLAIR.shape[4]
	filepath="best_model_supervised_FLAIR_to_T1.h5"
	model=load_UNET3D_Sintesis_v2(ps1,ps2,ps3,ch,nf)
	model.load_weights(filepath)
	FLAIR = np.reshape(FLAIR, (1,FLAIR.shape[0],FLAIR.shape[1],FLAIR.shape[2], 8))

	t1=time.time()
	res = TTDO(FLAIR,model)
	t2=time.time()
	print("tiempo=",t2-t1)
	res=res[0,0:181,0:217,0:181,0]
	nii.Nifti1Image(res,nii.load(flair_name).affine ).to_filename(t1_name)


	return t1_name

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
	# print("%d peaks found." % (len(peaks)))

	# norm_vol = vol
	if contrast.lower() in ["t1", "mprage"]:
		peak = peaks[-1]
		# print("Peak found at %.4f for %s" % (peak, contrast))
		# norm_vol = vol/peak
		# norm_vol[norm_vol > 1.25] = 1.25
		# norm_vol = norm_vol/1.25
	elif contrast.lower() in ['t2', 'pd', 'flair', 'fl']:
		peak_height = np.amax(heights)
		idx = np.where(heights == peak_height)
		peak = peaks[idx]
		# print("Peak found at %.4f for %s" % (peak, contrast))
		# norm_vol = vol / peak
		# norm_vol[norm_vol > 3.5] = 3.5
		# norm_vol = norm_vol / 3.5
	else:
		print("Contrast must be either t1,t2,pd, or flair. You entered %s. Returning 0." % contrast)

	# return peak, norm_vol
	return peak

def segment_image(nbNN, ps, Weights_list, T1, FLAIR, FG, normalization="kde"):
	crop_bg = 4
	first_time=1
	out_name= T1.replace('_t1',"").replace('preprocessed_',"lesions_")
	T1_img = nii.load(T1)
	T1=T1_img.get_data()
	T1=T1.astype('float32')
	FLAIR_img = nii.load(FLAIR)
	FLAIR=FLAIR_img.get_data()
	FLAIR=FLAIR.astype('float32')
	if(FG==None):
		MASK=(1-(T1==0).astype('float32'))
	else:
		MASK_img= nii.load(FG)
		MASK=MASK_img.get_data()
	ind=np.where(MASK>0)
	indbg=np.where(MASK==0)
	#MASK_img=(nii.Nifti1Image(MASK, T1_img.affine ))

	T1[indbg] = 0
	FLAIR[indbg] = 0

	if(normalization=='kde'):
		peak = normalize_image(T1, 't1')
		T1=T1/peak
		peak = normalize_image(FLAIR, 'flair')
		FLAIR=FLAIR/peak

	else:
		m1=np.mean(T1[ind])
		s1=np.std(T1[ind])
		T1[ind]=(T1[ind]-m1)/s1
		m1=np.mean(FLAIR[ind])
		s1=np.std(FLAIR[ind])
		FLAIR[ind]=(FLAIR[ind]-m1)/s1


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
	#print(out_shape)
	output=np.zeros(out_shape,T1.dtype)
	acu=np.zeros(out_shape[0:3],T1.dtype)

	t0=time.time()
	ii=0 # Network ID

	for ii in range(0,75):
		# select tile
		T = np.reshape(pT1[ii], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
		F = np.reshape(pFLAIR[ii], (1,pFLAIR.shape[1],pFLAIR.shape[2],pFLAIR.shape[3], 1))
		T=np.concatenate((T,F), axis=4)

		nc=2
		nf=24
		#print(T.shape)
		ch = T.shape[4]

		if(first_time==1):
			model=modelos.load_UNET3D_SLANT27_v2_groupNorm(ps[0],ps[1],ps[2],ch,nc,nf,0)
			first_time=0
		model.load_weights(Weights_list[ii])
		patch = model.predict(T)
		update_final_seg_with_tile_position(ii, output ,acu, patch,nbNN,offset1,offset2,offset3,crop_bg )
		if(ii< nbNN[1]*nbNN[2]*(nbNN[0]//2)):
			sym_tile=get_symetric_tile(ii,[5,5,5])
			T = np.reshape(pT1[sym_tile], (1,pT1.shape[1],pT1.shape[2],pT1.shape[3], 1))
			F = np.reshape(pFLAIR[sym_tile], (1,pFLAIR.shape[1],pFLAIR.shape[2],pFLAIR.shape[3], 1))
			T=np.concatenate((T,F), axis=4)
			patch = model.predict(T[:,-1::-1,:,:])[:,-1::-1,:,:]
			update_final_seg_with_tile_position(sym_tile, output ,acu, patch ,nbNN,offset1,offset2,offset3,crop_bg )

	K.clear_session()
	first_time=1
	#reconstruct
	print("Reconstructing segmentation...")
	ind=np.where(acu==0)
	mask_ind = np.where(acu>0)
	acu[ind]=1
	SEG= np.argmax(output, axis=3)
	SEG= np.reshape(SEG, SEG.shape[0:3])

	new_dtype = np.uint8
	SEG=SEG.astype(new_dtype)
	SEG=np.reshape(SEG,(T1.shape))
	t3=time.time()
	SEG_mask = SEG*MASK
	SEG_mask=SEG_mask.astype(new_dtype)
	SEG_mask=np.reshape(SEG_mask,(T1.shape))

	print("Processing time=",t3-t0)
	#write result (save file)
	array_img = nii.Nifti1Image(SEG_mask, T1_img.affine)
	array_img.set_data_dtype(new_dtype)
	array_img.to_filename(out_name)
	return out_name

def segmentation(nbNN,ps, Weights_list, T1_list, FLAIR_list,  FG_list, normalization="kde"):
	lesions_list=[]
	for i in range(len(T1_list)):
		try:
			lesions_list.append(segment_image(nbNN, ps, Weights_list, T1_list[i], T1_list[i], FG_list[i], normalization="kde"))
		except:
			lesions_list.append(segment_image(nbNN, ps, Weights_list, T1_list[i], T1_list[i], None, normalization="kde"))

	return lesions_list



def update_final_seg_with_tile_position(tile_num, final_seg,accumulation_update, out_to_put_in,nbNN,offset1,offset2,offset3,crop_bg ):
	#if(len(out_to_put_in.shape)>4):
	#	out_to_put_in=out_to_put_in[-4:]
	out_to_put_in=out_to_put_in[0,:,:,:,:]
	pos,a= get_tile_pos(tile_num,nbNN)
	x= crop_bg+ offset1* pos[0][0]
	#print(out_to_put_in.shape[0])
	y= crop_bg+ offset2* pos[1][0]
	z= offset3* pos[2][0]
	out_shape= final_seg.shape
	xx = x+out_to_put_in.shape[0]
	if xx> out_shape[0]:
		xx = out_shape[0]
	yy = y+out_to_put_in.shape[1]
	if yy> out_shape[1]:
		yy = out_shape[1]
	zz = z+out_to_put_in.shape[2]
	if zz> out_shape[2]:
		zz = out_shape[2]
	final_seg[x:xx,y:yy,z:zz]=final_seg[x:xx,y:yy,z:zz]+ out_to_put_in[0:xx-x,0:yy-y,0:zz-z]
	accumulation_update[x:xx,y:yy,z:zz]=accumulation_update[x:xx,y:yy,z:zz]+1

def get_tile_pos(ii,nbNN):
	n=0
	a=np.zeros((nbNN[0],nbNN[1],nbNN[2]))
	for x in range(nbNN[0]):
		for y in range(nbNN[1]):
			for z in range(nbNN[2]):
				a[x,y,z]= n
				n=n+1

	ind=np.where(a == ii)
	return ind,a

def get_symetric_tile(ii,nbNN):
	ind,a=  get_tile_pos(ii,nbNN)
	vv=int(a[-1-ind[0],ind[1],ind[2]])
	return vv

def to_native(inputname,transform_name,reference_name):
	outputname= inputname.replace('mni','native')
	ants_bin='/opt/deeplesionbrain/Registration/antsApplyTransforms'
	command=ants_bin+' -d 3 -i ' + inputname+' -r ' + reference_name +' -o ' +outputname + ' -n MultiLabel[0.3,0] -t [' + transform_name +', 1]';
	print(str(command))
	os.system(str(command))
	return outputname

def to_native_t1(inputname,outputname,transform_name,reference_name):
	ants_bin='/opt/deeplesionbrain/Registration/antsApplyTransforms'
	command=ants_bin+' -d 3 -i ' + inputname+' -r ' + reference_name +' -o ' +outputname + ' -n BSpline -t [' + transform_name +', 1]';
	print(str(command))
	os.system(str(command))
	return outputname

def to_MNI(inputname,outputname,transform_name,reference_name):
	ants_bin='/opt/deeplesionbrain/Registration/antsApplyTransforms'
	command=ants_bin+' -d 3 -i ' + inputname+' -r ' + reference_name +' -o ' +outputname + ' -n BSpline -t ' + transform_name
	print(str(command))
	os.system(str(command))
	return outputname
