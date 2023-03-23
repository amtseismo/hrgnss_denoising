#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 13:24:00 2022

Compare model results

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

fig, ax = plt.subplots(num=0, nrows=1,ncols=3,figsize=(12,4.5),sharex=True)
snrbins=np.arange(bw,20,0.1)
colors = np.array([[67/256,0,152/256,1],[0.6,0.6,0.6,1],[152/256,0,67/256,1]])
fillcolors = np.array([[67/256,0,152/256,.2],[0.6,0.6,0.6,.2],[152/256,0,67/256,.2]])
count=0
for model in ['1','2','3']:  
    for norm_input in [True]:
        model_save_file="quickie_3comp_norm_input_vmodel"+model+"_"+str(fac)+"_8-18.tf"  
            
        with open(model_save_file[:-3]+".res", "rb") as f:
            [SNR, dSNR, CC1, dCC, EUC, dEUC, truePGD, noisyPGD, predictedPGD] = pickle.load(f) 
        CC2=CC1+dCC
        # # this filters out stuff with SNR of 0 b/c Xcorr of all zeros with anything else is zero
        # tmp=np.unique(np.where(SNR==np.nan)[0])
        # SNR=SNR[tmp,:]
        # CC=CC[tmp,:]
        
        # x, y=SNR[:,0], CC[:,0]
        # # fit an array of size [Ndim, Nsamples]
        # data = np.vstack([x, y])
        # kde = gaussian_kde(data)
        
        # # evaluate on a regular grid
        # xgrid = np.linspace(0, 10, 1000)
        # ygrid = np.linspace(0, 1, 100)
        # Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        # Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        
        # # Plot the result as an image
        # plt.imshow(Z.reshape(Xgrid.shape),
        #            origin='lower', aspect='auto',
        #            extent=[0, 10, 0, 1],
        #            cmap='Blues')
        # cb = plt.colorbar()
        # cb.set_label("density")
        
        # # print(len(CC))
        # ax[0].plot(SNR[:,0],CC2[:,0],'o',color=colors[count],alpha=0.1)
        # ax[1].plot(SNR[:,1],CC2[:,1],'o',color=colors[count],alpha=0.1)
        # ax[2].plot(SNR[:,2],CC2[:,2],'o',label=model_save_file[14:-3],color=colors[count],alpha=0.1)   
        
        medsnrs=np.zeros((3,len(snrbins)))
        low=np.zeros((3,len(snrbins)))
        high=np.zeros((3,len(snrbins)))
        dmedsnrs=np.zeros((3,len(snrbins)))
        dlow=np.zeros((3,len(snrbins)))
        dhigh=np.zeros((3,len(snrbins)))
        for ii, snrbin in enumerate(snrbins):
            if model=='1':
                medsnrs[0,ii]=np.percentile(CC1[np.where((SNR[:,0]>=snrbin-bw) & (SNR[:,0]<snrbin+bw)),0],medb)
                medsnrs[1,ii]=np.percentile(CC1[np.where((SNR[:,1]>=snrbin-bw) & (SNR[:,1]<snrbin+bw)),1],medb)
                medsnrs[2,ii]=np.percentile(CC1[np.where((SNR[:,2]>=snrbin-bw) & (SNR[:,2]<snrbin+bw)),2],medb)

                low[0,ii]=np.percentile(CC1[np.where((SNR[:,0]>=snrbin-bw) & (SNR[:,0]<snrbin+bw)),0],lowb)
                low[1,ii]=np.percentile(CC1[np.where((SNR[:,1]>=snrbin-bw) & (SNR[:,1]<snrbin+bw)),1],lowb)
                low[2,ii]=np.percentile(CC1[np.where((SNR[:,2]>=snrbin-bw) & (SNR[:,2]<snrbin+bw)),2],lowb)

                high[0,ii]=np.percentile(CC1[np.where((SNR[:,0]>=snrbin-bw) & (SNR[:,0]<snrbin+bw)),0],highb)
                high[1,ii]=np.percentile(CC1[np.where((SNR[:,1]>=snrbin-bw) & (SNR[:,1]<snrbin+bw)),1],highb)
                high[2,ii]=np.percentile(CC1[np.where((SNR[:,2]>=snrbin-bw) & (SNR[:,2]<snrbin+bw)),2],highb)
            
            dmedsnrs[0,ii]=np.percentile(CC2[np.where((SNR[:,0]>=snrbin-bw) & (SNR[:,0]<snrbin+bw)),0],medb)
            dmedsnrs[1,ii]=np.percentile(CC2[np.where((SNR[:,1]>=snrbin-bw) & (SNR[:,1]<snrbin+bw)),1],medb)
            dmedsnrs[2,ii]=np.percentile(CC2[np.where((SNR[:,2]>=snrbin-bw) & (SNR[:,2]<snrbin+bw)),2],medb)

            dlow[0,ii]=np.percentile(CC2[np.where((SNR[:,0]>=snrbin-bw) & (SNR[:,0]<snrbin+bw)),0],lowb)
            dlow[1,ii]=np.percentile(CC2[np.where((SNR[:,1]>=snrbin-bw) & (SNR[:,1]<snrbin+bw)),1],lowb)
            dlow[2,ii]=np.percentile(CC2[np.where((SNR[:,2]>=snrbin-bw) & (SNR[:,2]<snrbin+bw)),2],lowb)

            dhigh[0,ii]=np.percentile(CC2[np.where((SNR[:,0]>=snrbin-bw) & (SNR[:,0]<snrbin+bw)),0],highb)
            dhigh[1,ii]=np.percentile(CC2[np.where((SNR[:,1]>=snrbin-bw) & (SNR[:,1]<snrbin+bw)),1],highb)
            dhigh[2,ii]=np.percentile(CC2[np.where((SNR[:,2]>=snrbin-bw) & (SNR[:,2]<snrbin+bw)),2],highb)
            
        if model=='1':
            #ax[0,0].plot(snrbins,medsnrs[0,:],linestyle='solid',linewidth=2,color='k')
            ax[0].plot(snrbins,medsnrs[0,:],linestyle='solid',linewidth=2,color='k')
            ax[1].plot(snrbins,medsnrs[1,:],linestyle='solid',linewidth=2,color='k')
            ax[2].plot(snrbins,medsnrs[2,:],linestyle='solid',linewidth=2,label="original",color='k')
            
        #ax[0,0].plot(snrbins,dmedsnrs[0,:],linestyle='dashed',linewidth=2,color=colors[count])
        ax[0].plot(snrbins,dmedsnrs[0,:],linestyle='dashed',linewidth=2,color=colors[count])
        ax[1].plot(snrbins,dmedsnrs[1,:],linestyle='dashed',linewidth=2,color=colors[count])
        ax[2].plot(snrbins,dmedsnrs[2,:],linestyle='dashed',linewidth=2,label="Model "+model,color=colors[count])
        
        
        ax[0].fill_between(snrbins,dlow[0,:],dhigh[0,:],facecolor=colors[count],edgecolor=fillcolors[count],alpha=0.15,linewidth=1)
        ax[1].fill_between(snrbins,dlow[1,:],dhigh[1,:],facecolor=colors[count],edgecolor=fillcolors[count],alpha=0.15,linewidth=1)
        ax[2].fill_between(snrbins,dlow[2,:],dhigh[2,:],facecolor=colors[count],edgecolor=fillcolors[count],alpha=0.15,linewidth=1)            
        count+=1
    print(str(np.average([np.interp(1,snrbins,dmedsnrs[0,:]),
                          np.interp(1,snrbins,dmedsnrs[1,:]),
                          np.interp(1,snrbins,dmedsnrs[2,:])])))

#ax[0,0].set_title('All',fontsize=14)
ax[0].set_title('North',fontsize=14)
ax[1].set_title('East',fontsize=14)
ax[2].set_title('Vertical',fontsize=14)

ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[2].tick_params(axis='both', which='major', labelsize=12)

maxlim=6
ax[0].set_xlabel('SNR',fontsize=14)
ax[0].set_ylabel('CC',fontsize=14)
ax[0].set_xlim((0,maxlim))
ax[0].set_ylim((-0.2,1))
ax[1].set_xlabel('SNR',fontsize=14)
ax[1].set_xlim((0,maxlim))
ax[1].set_ylim((-0.2,1))
ax[2].set_xlabel('SNR',fontsize=14)
ax[2].set_xlim((0,maxlim))
ax[2].set_ylim((-0.2,1))


plt.legend(loc="lower right",fontsize=14) #bbox_to_anchor=(0.6, 0.5))  
plt.tight_layout()
fig.savefig("cc_vs_snr.png",dpi=300)