#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:51:40 2022

Make plots of denoising examples

@author: amt
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import gnss_tools

sr=1
nperseg=31
noverlap=30
nlen=128
model='model3'
fac=2

if model=='model2':
    # Getting back the objects:
    with open('model2_v'+str(fac)+'_results.pkl','rb') as f:  # Python 3: open(..., 'rb')
        x, y, sigs, noise, x1, test_predictions = pickle.load(f)
    figinds=[8445, 8535, 16]
if model=='model3':
    # Getting back the objects:
    with open('model3_v'+str(fac)+'_results.pkl','rb') as f:  # Python 3: open(..., 'rb')
        x, y, sigs, noise, x1, test_predictions = pickle.load(f)
    figinds=[4100, 4721,20225]
    
f, t, Zxx = signal.stft(sigs[0,:256], fs=sr, nperseg=nperseg, noverlap=noverlap)

cmap='PuRd'
maxrange=x.shape[0]
lowamp=True
if lowamp:
    minval=0.01
    maxval=0.04
else:
    minval=0.1
    maxval=0.4

for count, ind in enumerate(figinds): 
    print(np.max(np.abs(sigs[ind,:])))
    if (np.max(np.abs(sigs[ind,:])) < maxval) and (np.max(np.abs(sigs[ind,:])) > minval):
        if lowamp:
            lim=mlim=0.04
            SNRmax=0
        else: 
            lim=mlim=0.4
        fig, axs = plt.subplots(num=ind, nrows=1,ncols=3,figsize=(12,3.5),sharex=True)
        t=np.arange(x[0,:,:,0].shape[1])*sr
        # plot original data and noise
        for comp in range(3):

            if comp==0:
                axs[comp].set_title('North')
                axs[comp].text(5,0.7*lim,'A',fontsize=16,weight='bold')
                axs[comp].set_ylabel('Amplitude (m)',fontsize=12)
            if comp==1:
                axs[comp].set_title('East')
                axs[comp].text(5,0.7*lim,'B',fontsize=16,weight='bold')
            if comp==2:
                axs[comp].set_title('Vertical')
                axs[comp].text(5,0.7*lim,'C',fontsize=16,weight='bold')
                for nn in range(3):
                    axs[nn].set_xlabel('Time (s)',fontsize=12)
                    axs[nn].set_xlim(0,128)
            
            # apply masks to noisy input signal and inverse transform 
            # note
            if model=='model2':
                _,tru_sig_inv=signal.istft(((y[ind,:,:,comp*2]+y[ind,:,:,comp*2+1]*1j)*np.max(np.abs(x1[ind,:,:,2*comp]+x1[ind,:,:,2*comp+1]*1j))), fs=sr, nperseg=nperseg, noverlap=noverlap)  
                _,est_sig_inv=signal.istft((test_predictions[ind,:,:,comp*2]+test_predictions[ind,:,:,comp*2+1]*1j)*np.max(np.abs(x1[ind,:,:,2*comp]+x1[ind,:,:,2*comp+1]*1j)), fs=sr, nperseg=nperseg, noverlap=noverlap)
            
            elif model=='model3':
                eps=1e-9
                sig_noise_stft=x1[ind,:,:,2*comp]+x1[ind,:,:,2*comp+1]*1j
                A=np.power(np.e,y[ind,:,:,comp*3])-eps
                costheta=y[ind,:,:,comp*3+1]
                sintheta=y[ind,:,:,comp*3+2]
                _,tru_sig_inv=signal.istft(sig_noise_stft*A*(costheta+sintheta*1j), 
                                                 fs=sr, nperseg=nperseg, noverlap=noverlap)
                A=np.power(np.e,test_predictions[ind,:,:,comp*3])-eps
                costheta=test_predictions[ind,:,:,comp*3+1]
                sintheta=test_predictions[ind,:,:,comp*3+2]
                _,est_sig_inv=signal.istft(sig_noise_stft*A*(costheta+sintheta*1j), 
                                                 fs=sr, nperseg=nperseg, noverlap=noverlap)
        
            true_noise=noise[ind,comp*nlen:(comp+1)*nlen]
            true_signal=sigs[ind,comp*nlen:(comp+1)*nlen]
            #inds=np.where(np.abs(true_signal)>0.00001)[0]
            #VR=gnss_tools.comp_VR(true_signal,est_sig_inv)    
            CC1, CC2=gnss_tools.comp_CC(true_signal, true_signal+true_noise, est_sig_inv)      
            SNR1, SNR2=gnss_tools.comp_SNR(true_signal, true_signal+true_noise, est_sig_inv)
            EUC1, EUC2=gnss_tools.comp_euclidean(true_signal, true_signal+true_noise, est_sig_inv)  
            if lowamp:
                if SNR1 > SNRmax:
                    SNRmax=SNR1

            axs[comp].plot(t,sigs[ind,comp*nlen:(comp+1)*nlen]+noise[ind,comp*nlen:(comp+1)*nlen], color=(0.6,0.6,0.6), label='signal+noise')        
            axs[comp].plot(t, true_signal, color=(67/256,0,152/256), label='signal')
            # axs[4,comp].plot(t, tru_sig_inv, alpha=0.75, color=(0.6,0,0), label='reconstructed')
            axs[comp].plot(t, est_sig_inv, color=(152/256,0,67/256), label='denoised signal')
            axs[comp].text(4,-0.9*lim,'CC='+'{:.3f}'.format(np.round(CC2[0]*1000)/1000), ha='left', va='bottom')
            axs[comp].text(123,-0.9*lim,"$\Delta$SNR="+'{:.1f}'.format(np.round((SNR2-SNR1)*100)/100), ha='right', va='bottom')
            axs[comp].text(45,-0.9*mlim,"SNR="+'{:.2f}'.format(np.round(SNR1*100)/100), va='bottom')
            axs[comp].text(123,0.9*lim,'$L^2$='+'{:.3f}'.format(np.round(EUC2*1000)/1000), ha='right', va='top')
            axs[comp].set_ylim((-lim,lim))
                
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        axs[comp].legend(bbox_to_anchor=(0.075, -0.15),ncol=3,frameon=False,fontsize=12)
        if lowamp and SNRmax>0 and SNRmax < 2:
            if model=='model2':
                fig.savefig("exfig_m2_"+str(ind)+".png", dpi=300)
            if model=='model3':
                fig.savefig("exfig_m3_"+str(ind)+".png", dpi=300)         
        elif ~lowamp:
            if model=='model2':
                fig.savefig("exfig_m2_"+str(ind)+".png", dpi=300)
            if model=='model3':
                fig.savefig("exfig_m3_"+str(ind)+".png", dpi=300)      