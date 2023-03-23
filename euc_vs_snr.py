#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 08:56:23 2022

Plot euclidean difference vs. snr

@author: amt
"""

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pickle
from matplotlib import cm
from scipy.stats import gaussian_kde

sr=1
eps=1e-9
nperseg=31
noverlap=30
norm_input=True
maskspercomp=1
bw=0.25
fac=2
lowb, medb, highb=10,50,90

fig, ax = plt.subplots(num=0, nrows=1,ncols=3,figsize=(12,4),sharex=True)
snrbins=np.arange(bw,20,0.1)
colors = np.array([[67/256,0,152/256,1],[0.6,0.6,0.6,1],[152/256,0,67/256,1]])
fillcolors = np.array([[67/256,0,152/256,.2],[0.6,0.6,0.6,.2],[152/256,0,67/256,.2]])
count=0
for model in ['1','2','3']: #,'2','3']:
    for norm_input in [True]:
        model_save_file="quickie_3comp_norm_input_vmodel"+model+"_"+str(fac)+"_8-18.tf"  
            
        with open(model_save_file[:-3]+".res", "rb") as f:
            [SNR, dSNR, CC, dCC, EUC, dEUC, truePGD, noisyPGD, predictedPGD] = pickle.load(f) 
        dEUC+=EUC
        print(np.max(np.abs(EUC)))
        print(np.max(np.abs(dEUC)))
        
        medsnrs=np.zeros((3,len(snrbins)))
        low=np.zeros((3,len(snrbins)))
        high=np.zeros((3,len(snrbins)))
        dmedsnrs=np.zeros((3,len(snrbins)))
        dlow=np.zeros((3,len(snrbins)))
        dhigh=np.zeros((3,len(snrbins)))
        for ii, snrbin in enumerate(snrbins):
            if model=='1':

                medsnrs[0,ii]=np.percentile(EUC[np.where((SNR[:,0]>=snrbin-bw) & (SNR[:,0]<snrbin+bw)),0],medb)
                medsnrs[1,ii]=np.percentile(EUC[np.where((SNR[:,1]>=snrbin-bw) & (SNR[:,1]<snrbin+bw)),1],medb)
                medsnrs[2,ii]=np.percentile(EUC[np.where((SNR[:,2]>=snrbin-bw) & (SNR[:,2]<snrbin+bw)),2],medb)

                low[0,ii]=np.percentile(EUC[np.where((SNR[:,0]>=snrbin-bw) & (SNR[:,0]<snrbin+bw)),0],lowb)
                low[1,ii]=np.percentile(EUC[np.where((SNR[:,1]>=snrbin-bw) & (SNR[:,1]<snrbin+bw)),1],lowb)
                low[2,ii]=np.percentile(EUC[np.where((SNR[:,2]>=snrbin-bw) & (SNR[:,2]<snrbin+bw)),2],lowb)

                high[0,ii]=np.percentile(EUC[np.where((SNR[:,0]>=snrbin-bw) & (SNR[:,0]<snrbin+bw)),0],highb)
                high[1,ii]=np.percentile(EUC[np.where((SNR[:,1]>=snrbin-bw) & (SNR[:,1]<snrbin+bw)),1],highb)
                high[2,ii]=np.percentile(EUC[np.where((SNR[:,2]>=snrbin-bw) & (SNR[:,2]<snrbin+bw)),2],highb)

            dmedsnrs[0,ii]=np.percentile(dEUC[np.where((SNR[:,0]>=snrbin-bw) & (SNR[:,0]<snrbin+bw)),0],medb)
            dmedsnrs[1,ii]=np.percentile(dEUC[np.where((SNR[:,1]>=snrbin-bw) & (SNR[:,1]<snrbin+bw)),1],medb)
            dmedsnrs[2,ii]=np.percentile(dEUC[np.where((SNR[:,2]>=snrbin-bw) & (SNR[:,2]<snrbin+bw)),2],medb)

            dlow[0,ii]=np.percentile(dEUC[np.where((SNR[:,0]>=snrbin-bw) & (SNR[:,0]<snrbin+bw)),0],lowb)
            dlow[1,ii]=np.percentile(dEUC[np.where((SNR[:,1]>=snrbin-bw) & (SNR[:,1]<snrbin+bw)),1],lowb)
            dlow[2,ii]=np.percentile(dEUC[np.where((SNR[:,2]>=snrbin-bw) & (SNR[:,2]<snrbin+bw)),2],lowb)

            dhigh[0,ii]=np.percentile(dEUC[np.where((SNR[:,0]>=snrbin-bw) & (SNR[:,0]<snrbin+bw)),0],highb)
            dhigh[1,ii]=np.percentile(dEUC[np.where((SNR[:,1]>=snrbin-bw) & (SNR[:,1]<snrbin+bw)),1],highb)
            dhigh[2,ii]=np.percentile(dEUC[np.where((SNR[:,2]>=snrbin-bw) & (SNR[:,2]<snrbin+bw)),2],highb)
            
        if model=='1':
            ax[0].plot(snrbins,medsnrs[0,:],linestyle='solid',linewidth=2,label="original",color='k')
            ax[1].plot(snrbins,medsnrs[1,:],linestyle='solid',linewidth=2,color='k')
            ax[2].plot(snrbins,medsnrs[2,:],linestyle='solid',linewidth=2,color='k')
        ax[0].plot(snrbins,dmedsnrs[0,:],linestyle='dashed',linewidth=2,label="Model "+model,color=colors[count])
        ax[1].plot(snrbins,dmedsnrs[1,:],linestyle='dashed',linewidth=2,color=colors[count])
        ax[2].plot(snrbins,dmedsnrs[2,:],linestyle='dashed',linewidth=2,color=colors[count])
        ax[0].fill_between(snrbins,dlow[0,:],dhigh[0,:],facecolor=colors[count],edgecolor=fillcolors[count],alpha=0.1,linewidth=1)
        ax[1].fill_between(snrbins,dlow[1,:],dhigh[1,:],facecolor=colors[count],edgecolor=fillcolors[count],alpha=0.1,linewidth=1)
        ax[2].fill_between(snrbins,dlow[2,:],dhigh[2,:],facecolor=colors[count],edgecolor=fillcolors[count],alpha=0.1,linewidth=1)            
        count+=1

ax[0].set_title('North',fontsize=14)
ax[1].set_title('East',fontsize=14)
ax[2].set_title('Vertical',fontsize=14)

ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[2].tick_params(axis='both', which='major', labelsize=12)

EUCmin=np.min(np.concatenate((EUC,dEUC)))
EUCmax=np.max(np.concatenate((EUC,dEUC)))
#ax[0,0].set_title('All',fontsize=14)
ax[0].set_title('North',fontsize=14)
ax[1].set_title('East',fontsize=14)
ax[2].set_title('Vertical',fontsize=14)

ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[2].tick_params(axis='both', which='major', labelsize=12)

maxsnr=6
maxeuc=0.12
ax[0].set_xlabel('SNR',fontsize=14)
ax[0].set_ylabel('$L^2$ (m)',fontsize=14)
ax[0].set_xlim((0,maxsnr))
ax[0].set_ylim((0,maxeuc))
ax[1].set_xlabel('SNR',fontsize=14)
ax[1].set_xlim((0,maxsnr))
ax[1].set_ylim((0,maxeuc))
ax[2].set_xlabel('SNR',fontsize=14)
ax[2].set_xlim((0,maxsnr))
ax[2].set_ylim((0,maxeuc))


ax[0].legend(loc="upper left",fontsize=14) #bbox_to_anchor=(0.6, 0.5))  
plt.tight_layout()
fig.savefig("euc_vs_snr.png", dpi=300)