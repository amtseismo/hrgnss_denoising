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
model='model1'
fac=2
np.random.seed(0)

if model=='model1':
    # Getting back the objects:
    with open('model1_v'+str(fac)+'_results.pkl','rb') as f:  # Python 3: open(..., 'rb')
        x, y, sigs, noise, x1, test_predictions = pickle.load(f)
plt.figure()
plt.plot(sigs[0,:])

    
f, t, Zxx = signal.stft(sigs[0,:256], fs=sr, nperseg=nperseg, noverlap=noverlap)

cmap='PuRd'
maxrange=x.shape[0]
lowamp=False
if lowamp:
    minval=0.01
    maxval=0.03
else:
    minval=0.1
    maxval=2
    
for count, ind in enumerate([1187, 2163, 46]): # LSNR - 2163, 1187 HSNR - 46
    #print(str(np.max(np.abs(sigs[ind,:]))))
    print(ind)
    if (np.max(np.abs(sigs[ind,:])) < maxval) and (np.max(np.abs(sigs[ind,:])) > minval):
        if lowamp:
            lim=mlim=0.04
            SNRmax=0
        else: 
            lim=mlim=0.4
        #print(str((np.max(np.abs(sigs[ind,:])))))
        fig, axs = plt.subplots(num=ind, nrows=3,ncols=4,figsize=(12,7),sharex=True)
        t=np.arange(x[0,:,:,0].shape[1])*sr
        # plot original data and noise
        for comp in range(3):
            axs[comp,0].plot(t,sigs[ind,comp*nlen:(comp+1)*nlen]+noise[ind,comp*nlen:(comp+1)*nlen], color=(0.6,0.6,0.6), label='signal+noise')
            axs[comp,0].plot(t,sigs[ind,comp*nlen:(comp+1)*nlen], color=(67/256,0,152/256), label='signal')
            axs[comp,0].set_ylabel('Amplitude (m)',fontsize=12)
            axs[comp,1].set_ylabel('Frequency (hz)',fontsize=12)
            axs[comp,2].set_ylabel('Frequency (hz)',fontsize=12)
            axs[comp,3].set_ylabel('Amplitude (m)',fontsize=12)
            if comp==0:
                axs[comp,0].set_title('North')
                axs[comp,0].text(5,0.7*lim,'A',fontsize=16,weight='bold')
                axs[comp,1].set_title('North mask (target)')
                axs[comp,1].text(5,0.41,'B',fontsize=16,weight='bold')
                axs[comp,2].set_title('North mask (predicted)')
                axs[comp,2].text(5,0.41,'C',fontsize=16,weight='bold')
                axs[comp,3].set_title('North (denoised)')
                axs[comp,3].text(5,0.7*lim,'D',fontsize=16,weight='bold')
            if comp==1:
                axs[comp,0].set_title('East')
                axs[comp,0].text(5,0.7*lim,'E',fontsize=16,weight='bold')
                axs[comp,1].set_title('East mask (target)')
                axs[comp,1].text(5,0.41,'F',fontsize=16,weight='bold')
                axs[comp,2].set_title('East mask (predicted)')
                axs[comp,2].text(5,0.41,'G',fontsize=16,weight='bold')
                axs[comp,3].set_title('East (denoised)')
                axs[comp,3].text(5,0.7*lim,'H',fontsize=16,weight='bold')
            if comp==2:
                axs[comp,0].set_title('Vertical')
                axs[comp,0].text(5,0.7*lim,'I',fontsize=16,weight='bold')
                axs[comp,1].set_title('Vertical mask (target)')
                axs[comp,1].text(5,0.41,'J',fontsize=16,weight='bold')
                axs[comp,2].set_title('Vertical mask (predicted)')
                axs[comp,2].text(5,0.41,'K',fontsize=16,weight='bold')
                axs[comp,3].set_title('Vertical (denoised)')
                axs[comp,3].text(5,0.7*lim,'L',fontsize=16,weight='bold')
                for nn in range(4):
                    axs[comp,nn].set_xlabel('Time (s)',fontsize=12)
            axs[comp,3].set_ylabel('Amplitude (m)',fontsize=12)
        # axs[0,comp*2+1].plot(t,noise[ind,comp*nlen:(comp+1)*nlen], label='noise')
        # axs[0,comp*2+1].plot(t,sigs[ind,comp*nlen:(comp+1)*nlen]+noise[ind,comp*nlen:(comp+1)*nlen], color=(0.6,0.6,0.6), alpha=0.5, label='signal+noise')
        # # axs[0,comp*2+1].legend()
        # axs[0,comp*2+1].set_title('Original noise')   
            
            axs[comp,0].set_ylim((-lim,lim))
            
            # plot the output signal and noise masks
            axs[comp,1].pcolormesh(t, f, y[ind,:,:,comp], cmap=cmap, vmin=0, vmax=1, shading='auto')
            
            #noise mask predicted
            pcm=axs[comp,2].pcolormesh(t, f, test_predictions[ind,:,:,comp], cmap=cmap, vmin=0, vmax=1, shading='auto')
        
            # apply masks to noisy input signal and inverse transform 
            if model=='model1':
                _,tru_sig_inv=signal.istft(y[ind,:,:,comp]*(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
                _,tru_noise_inv=signal.istft((1-y[ind,:,:,comp])*(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)    
                _,est_sig_inv=signal.istft(test_predictions[ind,:,:,comp]*(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
                _,est_noise_inv=signal.istft((1-test_predictions[ind,:,:,comp])*(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap) 
    
            true_noise=noise[ind,comp*nlen:(comp+1)*nlen]
            true_signal=sigs[ind,comp*nlen:(comp+1)*nlen]
            #inds=np.where(np.abs(true_signal)>0.00001)[0]
            #VR=gnss_tools.comp_VR(true_signal,est_sig_inv)    
            # print(true_signal[:5])
            # print(true_noise[:5])
            # print(est_sig_inv[:5])
            CC1, CC2=gnss_tools.comp_CC(true_signal, true_signal+true_noise, est_sig_inv)      
            SNR1, SNR2=gnss_tools.comp_SNR(true_signal, true_signal+true_noise, est_sig_inv)
            EUC1, EUC2=gnss_tools.comp_euclidean(true_signal, true_signal+true_noise, est_sig_inv) 

            axs[comp,3].plot(t,sigs[ind,comp*nlen:(comp+1)*nlen]+noise[ind,comp*nlen:(comp+1)*nlen], color=(0.6,0.6,0.6), label='signal+noise')        
            axs[comp,3].plot(t, true_signal, color=(67/256,0,152/256), label='signal')
            # axs[4,comp].plot(t, tru_sig_inv, alpha=0.75, color=(0.6,0,0), label='reconstructed')
            axs[comp,3].plot(t, est_sig_inv, color=(152/256,0,67/256), label='denoised signal')
            axs[comp,3].text(4,-0.9*lim,'CC='+'{:.3f}'.format(np.round(CC2[0]*1000)/1000), ha='left', va='bottom')
            axs[comp,3].text(123,0.9*lim,'$L^2$='+'{:.3f}'.format(np.round(EUC2*1000)/1000), ha='right', va='top')
            axs[comp,3].text(123,-0.9*lim,"$\Delta$SNR="+'{:.1f}'.format(np.round((SNR2-SNR1)*100)/100), ha='right', va='bottom')
            axs[comp,0].text(123,-0.9*lim,"SNR="+'{:.1f}'.format(np.round(SNR1*100)/100), ha='right', va='bottom')
            #axs[comp,0].text(80,-0.7*lim,'$L^2$='+'{:.3f}'.format(np.round(EUC1*1000)/1000))
            axs[comp,3].set_ylim((-lim,lim))
            
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.15)
        cbar_ax = fig.add_axes([0.18, 0.04, 0.25, 0.024])
        cbar_ax.text(-.15,0,'M$_S$', fontsize=14)
        
        fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', shrink=0.1)
        axs[comp,3].legend(bbox_to_anchor=(0.4, -0.28),ncol=3,frameon=False,fontsize=12)
        fig.savefig("exfig_m1_"+str(ind)+".png", dpi=300)
        