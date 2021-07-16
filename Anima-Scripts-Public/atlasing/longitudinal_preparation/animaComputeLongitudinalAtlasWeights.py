#!/usr/bin/python3
# Warning: works only on unix-like systems, not windows where "python animaComputeLongitudinalAtlasWeights.py ..." has to be run

import argparse
import os
import shutil
import numpy as np
from scipy import signal
import pandas as pd 
from animaPolynomialKernel import polynomial_kernel

# Argument parsing
parser = argparse.ArgumentParser(description="Compute data weights for building an atlas at the specified age")

parser.add_argument('-a', '--age-file', required=True, type=str, help='list of ages (in txt file)')
parser.add_argument('-i', '--image-file', required=True, type=str, help='list of images (in txt file)')
parser.add_argument('-o', '--out-dir', required=True, type=str, help='output directory')
parser.add_argument('-n', '--nb-images', required=True, type=int, help='desired number of images per atlas')
parser.add_argument('-A', '--age-atlas', required=True, type=str, help='desired age of each sub-atlas')
parser.add_argument('-p', '--prefix', required=True, type=str, help='prefix of subjects')

parser.add_argument('-t', '--t-sampleSize', type=int, default=1000, help='size of age sampling: (default: 1000)')
parser.add_argument('-u', '--nb-iter', type=int, default=30, help='number of iteration of main loop (optimization) (default: 30)')
parser.add_argument('-v', '--alpha-sampleSize', type=int, default=500, help='size of alpha sampling (default: 500)')
parser.add_argument('-b', '--tol-bias', type=float, default=0.005, help='maximum bias tolerance (default: 0.005)')
parser.add_argument('-s', '--init-window', type=float, default=3, help='initial size of age window (default: 3)')

args = parser.parse_args()

ages = np.loadtxt(fname=args.age_file)
images = np.genfromtxt(args.image_file, dtype='str')
outDir = args.out_dir
N = args.nb_images
wantedAge = np.loadtxt(fname=args.age_atlas)
prefix = os.path.split(args.prefix)

sampleSize = args.t_sampleSize
itmax = args.nb_iter
alphaSampleSize = args.alpha_sampleSize
tolBias = args.tol_bias
s = args.init_window*np.ones(sampleSize)

t = np.linspace(min(ages), max(ages), sampleSize)
alpha = np.zeros(sampleSize)
n = np.zeros(sampleSize)

print("optimizing kernel window over temporal bias...")
for it in range(1, itmax+1):
    print(str(it)+"/"+str(itmax))
    
    bias = np.inf*np.ones(len(t))
    for i in range(0, len(t)):
        rangeAlpha = np.linspace(np.ceil(10000*(t[i]-3*s[i]/5))/10000, np.floor(10000*(t[i]-2*s[i]/5))/10000, alphaSampleSize)
    
        for alpha0 in rangeAlpha:
            _, _, bias0, n0 = polynomial_kernel(ages, t[i], s[i], alpha0)
            if bias0 < bias[i]:
                alpha[i] = alpha0
                bias[i] = bias0
                n[i] = n0
        
        if it < itmax:
            st=0.5*0.8**(it-1)        
            if n[i] < N:
                s[i] = s[i]+st
            elif n[i] > N:
                s[i] = s[i]-st
            
    if it == itmax-1:
        s = signal.savgol_filter(s, int(2*np.floor(sampleSize/20)+1), 3)
            
modelInfo = {'sampleTime': t, 'windowSize': s, 'windowStart': alpha, 'windowFrequency': n, 'temporalBias': bias}
df = pd.DataFrame(data=modelInfo)

if not os.path.exists(outDir):
    os.makedirs(outDir)

df.to_csv(os.path.join(outDir, "modelInfo.csv"))
 
print("choosing ages and subjects for each sub-atlas...")      
okAge = t[bias < tolBias]
atlasAge = np.zeros(len(wantedAge))

for i in range(0, len(wantedAge)):
    indAge = abs(wantedAge[i]-okAge).argmin()
    atlasAge[i] = okAge[indAge]

np.savetxt(os.path.join(outDir, "atlasAge.txt"), atlasAge)

print("mkdirs and cp files...")
for i in range(0, len(atlasAge)): 
    indt = np.where(t == atlasAge[i])
    w, ind, _, _ = polynomial_kernel(ages, t[indt], s[indt], alpha[indt])
    sub=images[ind]

    if os.path.exists(os.path.join(outDir, "atlas_"+str(i+1))):
        shutil.rmtree(os.path.join(outDir, "atlas_"+str(i+1)))

    os.makedirs(os.path.join(outDir, "atlas_"+str(i+1)))
    os.makedirs(os.path.join(outDir, "atlas_"+str(i+1), prefix[0]))
    for j in range(0, len(sub)):
        fileExtension = os.path.splitext(sub[j])[1]
        if fileExtension == '.gz':
            fileExtension = os.path.splitext(os.path.splitext(sub[j])[0])[1] + fileExtension

        dest = os.path.join(outDir, "atlas_"+str(i+1), prefix[0], prefix[1]+"_"+str(j+1)+fileExtension)
        shutil.copyfile(sub[j], str(dest))
        
    np.savetxt(os.path.join(outDir, "atlas_"+str(i+1),"weights.txt"),w)
    np.savetxt(os.path.join(outDir, "atlas_"+str(i+1),"subjects.txt"),sub, fmt="%s")
