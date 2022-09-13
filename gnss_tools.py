#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:08:55 2020

Unet models

@author: amt
"""

import tensorflow as tf
import numpy as np
from scipy import signal
from obspy.signal.cross_correlation import correlate_template
#import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import pydot             

def stft_3comp_data_generator(model, batch_size, x_data, n_data, sr, nperseg, noverlap, norm_input, eps=1e-9, nlen=128, valid=False):
    while True:
        # grab batch
        sig=x_data[np.random.choice(len(x_data),batch_size),:]
        noise=n_data[np.random.choice(len(n_data),batch_size),:]
        # calculate shifts
        if not valid:
            stime_offset=np.random.uniform(0,nlen,size=batch_size)                     
        else:
            stime_offset=nlen//2*np.ones(batch_size)           
        ntime_offset=np.random.uniform(0,nlen,size=batch_size)
        # get er done
        batch_inputs=np.zeros((batch_size,16,nlen,6))
        raw_stft=np.zeros((batch_size,16,nlen,6))
        if model=='v1':
            batch_outputs=np.zeros((batch_size,16,nlen,3))
        elif model=='v2':
            batch_outputs=np.zeros((batch_size,16,nlen,6))
        elif model=='v3':
            batch_outputs=np.zeros((batch_size,16,nlen,9))
        subsigs=np.zeros((len(stime_offset),3*nlen))
        subnoises=np.zeros((len(stime_offset),3*nlen))
        for ii in range(len(stime_offset)):
            # window data, noise, and data+noise timeseries based on shift
            si1=int(stime_offset[ii]*sr)
            si2=int(stime_offset[ii]*sr)+int(nlen*sr)
            si3=si1+nlen*2
            si4=si2+nlen*2
            si5=si1+nlen*4
            si6=si2+nlen*4
            subsig1=sig[ii,si1:si2]
            subsig2=sig[ii,si3:si4]
            subsig3=sig[ii,si5:si6]
            ni1=int(ntime_offset[ii]*sr)
            ni2=int(ntime_offset[ii]*sr)+int(nlen*sr)
            ni3=ni1+nlen*2
            ni4=ni2+nlen*2
            ni5=ni1+nlen*4
            ni6=ni2+nlen*4
            subnoise1=noise[ii,ni1:ni2]
            subnoise2=noise[ii,ni3:ni4]
            subnoise3=noise[ii,ni5:ni6]
            subsigs[ii,:]=np.concatenate((subsig1,subsig2,subsig3))
            subnoises[ii,:]=np.concatenate((subnoise1,subnoise2,subnoise3))
            _, _, stftsig1 = signal.stft(subsig1, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise1 = signal.stft(subnoise1, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput1 = stftsig1+stftnoise1
            _, _, stftsig2 = signal.stft(subsig2, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise2 = signal.stft(subnoise2, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput2 = stftsig2+stftnoise2
            _, _, stftsig3 = signal.stft(subsig3, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise3 = signal.stft(subnoise3, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput3 = stftsig3+stftnoise3
            # batch inputs are the real and imaginary parts of the stft of signal+noise
            raw_stft[ii,:,:,:]=data_2_input(stftinput1,stftinput2,stftinput3,norm_input=False)
            batch_inputs[ii,:,:,:]=data_2_input(stftinput1,stftinput2,stftinput3,norm_input)
            if model=='v1':
                # batch outputs are real valued signal masks
                with np.errstate(divide='ignore'):
                    rat1=np.nan_to_num(np.abs(stftnoise1)/np.abs(stftsig1),posinf=1e20)
                batch_outputs[ii,:,:,0]=1/(1+rat1) # signal mask
                # batch outputs are 
                with np.errstate(divide='ignore'):
                    rat2=np.nan_to_num(np.abs(stftnoise2)/np.abs(stftsig2),posinf=1e20)
                batch_outputs[ii,:,:,1]=1/(1+rat2) # signal mask
                # batch outputs are 
                with np.errstate(divide='ignore'):
                    rat3=np.nan_to_num(np.abs(stftnoise3)/np.abs(stftsig3),posinf=1e20)
                batch_outputs[ii,:,:,2]=1/(1+rat3) # signal mask
            elif model=='v2':
                # batch outputs are 
                batch_outputs[ii,:,:,0]=np.real(stftsig1)
                batch_outputs[ii,:,:,1]=np.imag(stftsig1)
                batch_outputs[ii,:,:,2]=np.real(stftsig2)
                batch_outputs[ii,:,:,3]=np.imag(stftsig2)
                batch_outputs[ii,:,:,4]=np.real(stftsig3)
                batch_outputs[ii,:,:,5]=np.imag(stftsig3)
                if norm_input:
                    batch_outputs[ii,:,:,0]=batch_outputs[ii,:,:,0]/np.max(np.abs(stftinput1))
                    batch_outputs[ii,:,:,1]=batch_outputs[ii,:,:,1]/np.max(np.abs(stftinput1))
                    batch_outputs[ii,:,:,2]=batch_outputs[ii,:,:,2]/np.max(np.abs(stftinput2))
                    batch_outputs[ii,:,:,3]=batch_outputs[ii,:,:,3]/np.max(np.abs(stftinput2))
                    batch_outputs[ii,:,:,4]=batch_outputs[ii,:,:,4]/np.max(np.abs(stftinput3))
                    batch_outputs[ii,:,:,5]=batch_outputs[ii,:,:,5]/np.max(np.abs(stftinput3))
            elif model=='v3':
                # batch outputs are 
                batch_outputs[ii,:,:,0]=np.log(np.abs(stftsig1/stftinput1)+eps)
                batch_outputs[ii,:,:,1]=np.cos(np.angle(stftsig1/stftinput1))
                batch_outputs[ii,:,:,2]=np.sin(np.angle(stftsig1/stftinput1))
                batch_outputs[ii,:,:,3]=np.log(np.abs(stftsig2/stftinput2)+eps)
                batch_outputs[ii,:,:,4]=np.cos(np.angle(stftsig2/stftinput2))
                batch_outputs[ii,:,:,5]=np.sin(np.angle(stftsig2/stftinput2))
                batch_outputs[ii,:,:,6]=np.log(np.abs(stftsig3/stftinput3)+eps)
                batch_outputs[ii,:,:,7]=np.cos(np.angle(stftsig3/stftinput3))
                batch_outputs[ii,:,:,8]=np.sin(np.angle(stftsig3/stftinput3))
            # make all angles positive
        if valid:
            yield(batch_inputs,batch_outputs,subsigs,subnoises,raw_stft)
        else:
            yield(batch_inputs,batch_outputs)
   
def stft_3comp_data_processor(model, sigs, noise, sr, nperseg, noverlap, norm_input, eps=1e-9, nlen=128):
    while True:
        # get er done
        batch_inputs=np.zeros((sigs.shape[0],16,nlen,6))
        raw_stft=np.zeros((sigs.shape[0],16,nlen,6))
        if model=='v1':
            batch_outputs=np.zeros((sigs.shape[0],16,nlen,3))
        elif model=='v2':
            batch_outputs=np.zeros((sigs.shape[0],16,nlen,6))
        elif model=='v3':
            batch_outputs=np.zeros((sigs.shape[0],16,nlen,9))
        for ii in range(sigs.shape[0]):
            # window data, noise, and data+noise timeseries based on shift
            subsig1=sigs[ii,:nlen]
            subsig2=sigs[ii,nlen:2*nlen]
            subsig3=sigs[ii,2*nlen:3*nlen]
            subnoise1=noise[ii,:nlen]
            subnoise2=noise[ii,nlen:2*nlen]
            subnoise3=noise[ii,2*nlen:3*nlen]
            _, _, stftsig1 = signal.stft(subsig1, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise1 = signal.stft(subnoise1, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput1 = stftsig1+stftnoise1
            _, _, stftsig2 = signal.stft(subsig2, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise2 = signal.stft(subnoise2, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput2 = stftsig2+stftnoise2
            _, _, stftsig3 = signal.stft(subsig3, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise3 = signal.stft(subnoise3, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput3 = stftsig3+stftnoise3
            # batch inputs are the real and imaginary parts of the stft of signal+noise
            raw_stft[ii,:,:,:]=data_2_input(stftinput1,stftinput2,stftinput3,norm_input=False)
            batch_inputs[ii,:,:,:]=data_2_input(stftinput1,stftinput2,stftinput3,norm_input)
            if model=='v1':
                # batch outputs are real valued signal masks
                with np.errstate(divide='ignore'):
                    rat1=np.nan_to_num(np.abs(stftnoise1)/np.abs(stftsig1),posinf=1e20)
                batch_outputs[ii,:,:,0]=1/(1+rat1) # signal mask
                # batch outputs are 
                with np.errstate(divide='ignore'):
                    rat2=np.nan_to_num(np.abs(stftnoise2)/np.abs(stftsig2),posinf=1e20)
                batch_outputs[ii,:,:,1]=1/(1+rat2) # signal mask
                # batch outputs are 
                with np.errstate(divide='ignore'):
                    rat3=np.nan_to_num(np.abs(stftnoise3)/np.abs(stftsig3),posinf=1e20)
                batch_outputs[ii,:,:,2]=1/(1+rat3) # signal mask
            elif model=='v2':
                # batch outputs are 
                batch_outputs[ii,:,:,0]=np.real(stftsig1)
                batch_outputs[ii,:,:,1]=np.imag(stftsig1)
                batch_outputs[ii,:,:,2]=np.real(stftsig2)
                batch_outputs[ii,:,:,3]=np.imag(stftsig2)
                batch_outputs[ii,:,:,4]=np.real(stftsig3)
                batch_outputs[ii,:,:,5]=np.imag(stftsig3)
                if norm_input:
                    batch_outputs[ii,:,:,0]=batch_outputs[ii,:,:,0]/np.max(np.abs(stftinput1))
                    batch_outputs[ii,:,:,1]=batch_outputs[ii,:,:,1]/np.max(np.abs(stftinput1))
                    batch_outputs[ii,:,:,2]=batch_outputs[ii,:,:,2]/np.max(np.abs(stftinput2))
                    batch_outputs[ii,:,:,3]=batch_outputs[ii,:,:,3]/np.max(np.abs(stftinput2))
                    batch_outputs[ii,:,:,4]=batch_outputs[ii,:,:,4]/np.max(np.abs(stftinput3))
                    batch_outputs[ii,:,:,5]=batch_outputs[ii,:,:,5]/np.max(np.abs(stftinput3))
            elif model=='v3':
                # batch outputs are 
                batch_outputs[ii,:,:,0]=np.log(np.abs(stftsig1/stftinput1)+eps)
                batch_outputs[ii,:,:,1]=np.cos(np.angle(stftsig1/stftinput1))
                batch_outputs[ii,:,:,2]=np.sin(np.angle(stftsig1/stftinput1))
                batch_outputs[ii,:,:,3]=np.log(np.abs(stftsig2/stftinput2)+eps)
                batch_outputs[ii,:,:,4]=np.cos(np.angle(stftsig2/stftinput2))
                batch_outputs[ii,:,:,5]=np.sin(np.angle(stftsig2/stftinput2))
                batch_outputs[ii,:,:,6]=np.log(np.abs(stftsig3/stftinput3)+eps)
                batch_outputs[ii,:,:,7]=np.cos(np.angle(stftsig3/stftinput3))
                batch_outputs[ii,:,:,8]=np.sin(np.angle(stftsig3/stftinput3))
            # make all angles positive
        yield(batch_inputs,batch_outputs,raw_stft)
                 
def data_2_input(stftinput1,stftinput2,stftinput3,norm_input):
    if norm_input:
        stftinput1=stftinput1/np.max(np.abs(stftinput1))
        stftinput2=stftinput2/np.max(np.abs(stftinput2))
        stftinput3=stftinput3/np.max(np.abs(stftinput3))
    tmp=np.zeros((1,stftinput1.shape[0],stftinput1.shape[1],6))
    tmp[0,:,:,0]=np.real(stftinput1) 
    tmp[0,:,:,1]=np.imag(stftinput1)
    tmp[0,:,:,2]=np.real(stftinput2) 
    tmp[0,:,:,3]=np.imag(stftinput2) 
    tmp[0,:,:,4]=np.real(stftinput3) 
    tmp[0,:,:,5]=np.imag(stftinput3)
    return tmp

def output_2_data(model,y,x1,test_predictions,sr,nperseg,noverlap,norm_input,eps=1e-9):
    '''
    converts model and generator output to true signal and noise and 
    estimated signal and noise

    Parameters
    ----------
    model : string
        model version
    y : complex array of floats
        model target
    x1 : complex array of floats
        unnormalized STFT
    test_predictions : TYPE
        model output
    sr : float
        sample rate
    nperseg : int
        number per segment input into STFT
    noverlap : int
        noverlap input into STFT

    Returns
    -------
    tru_sig_inv : real array
        true signal
    tru_noise_inv : real array
        true noise
    est_sig_inv : real array
        model predicted signal
    est_noise_inv : real array
        model predicted noise

    '''
    nlen=y.shape[2]
    tru_sig_inv=np.zeros((y.shape[0],3*y.shape[2]))
    tru_noise_inv=np.zeros((y.shape[0],3*y.shape[2]))
    est_sig_inv=np.zeros((y.shape[0],3*y.shape[2]))
    est_noise_inv=np.zeros((y.shape[0],3*y.shape[2]))
    # note for model 2 and 3 you dont get a noise estimate
    for ind in np.arange(y.shape[0]):
        for comp in range(3):
            if model=='v1':
                # apply masks to noisy input signal and inverse transform 
                _,tru_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft(y[ind,:,:,comp]*(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
                _,tru_noise_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft((1-y[ind,:,:,comp])*(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)    
                _,est_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft(test_predictions[ind,:,:,comp]*(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
                _,est_noise_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft((1-test_predictions[ind,:,:,comp])*(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
            elif model=='v2':
                if not(norm_input):
                    _,tru_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft((y[ind,:,:,comp*2]+y[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)  
                    _,est_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft((test_predictions[ind,:,:,comp*2]+test_predictions[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
                else:
                    _,tru_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft(((y[ind,:,:,comp*2]+y[ind,:,:,comp*2+1]*1j)*np.max(np.abs(x1[ind,:,:,2*comp]+x1[ind,:,:,2*comp+1]*1j))), fs=sr, nperseg=nperseg, noverlap=noverlap)  
                    _,est_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft((test_predictions[ind,:,:,comp*2]+test_predictions[ind,:,:,comp*2+1]*1j)*np.max(np.abs(x1[ind,:,:,2*comp]+x1[ind,:,:,2*comp+1]*1j)), fs=sr, nperseg=nperseg, noverlap=noverlap)
                _,noisysig=signal.istft(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j, fs=sr, nperseg=nperseg, noverlap=noverlap)
                tru_noise_inv[ind,comp*nlen:(comp+1)*nlen]=noisysig-tru_sig_inv[ind,comp*nlen:(comp+1)*nlen]
                est_noise_inv[ind,comp*nlen:(comp+1)*nlen]=noisysig-est_sig_inv[ind,comp*nlen:(comp+1)*nlen]
            elif model=='v3':
                _,noisysig=signal.istft(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j, fs=sr, nperseg=nperseg, noverlap=noverlap)
                '''
                sig_noise_stft=x1[ind,:,:,2*comp]+x1[ind,:,:,2*comp+1]*1j
                A=np.power(np.e,y[ind,:,:,comp*3])-eps
                costheta=y[ind,:,:,comp*3+1]
                sintheta=y[ind,:,:,comp*3+2]
                _,tru_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft(sig_noise_stft*A*(costheta+sintheta*1j), 
                                                 fs=sr, nperseg=nperseg, noverlap=noverlap)
                A=np.power(np.e,test_predictions[ind,:,:,comp*3])-eps
                costheta=test_predictions[ind,:,:,comp*3+1]
                sintheta=test_predictions[ind,:,:,comp*3+2]
                _,est_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft(sig_noise_stft*A*(costheta+sintheta*1j), 
                                                 fs=sr, nperseg=nperseg, noverlap=noverlap) '''
                sig_noise_stft=x1[ind,:,:,2*comp]+x1[ind,:,:,2*comp+1]*1j
                _,tru_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft(sig_noise_stft*(np.power(np.e,y[ind,:,:,comp*3])-eps)*(y[ind,:,:,comp*3+1]+y[ind,:,:,comp*3+2]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
                _,est_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft(sig_noise_stft*(np.power(np.e,test_predictions[ind,:,:,comp*3])-eps)*(test_predictions[ind,:,:,comp*3+1]+test_predictions[ind,:,:,comp*3+2]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
                tru_noise_inv[ind,comp*nlen:(comp+1)*nlen]=noisysig-tru_sig_inv[ind,comp*nlen:(comp+1)*nlen]
                est_noise_inv[ind,comp*nlen:(comp+1)*nlen]=noisysig-est_sig_inv[ind,comp*nlen:(comp+1)*nlen]              
    return tru_sig_inv, tru_noise_inv, est_sig_inv, est_noise_inv

def real_output_2_data(model,x1,test_predictions,sr,nperseg,noverlap,norm_input,eps=1e-9):
    '''
    converts model and generator output to true signal and noise and 
    estimated signal and noise

    Parameters
    ----------
    model : string
        model version
    x1 : complex array of floats
        unnormalized STFT
    test_predictions : TYPE
        model output
    sr : float
        sample rate
    nperseg : int
        number per segment input into STFT
    noverlap : int
        noverlap input into STFT

    Returns
    -------
    est_sig_inv : real array
        model predicted signal

    '''
    nlen=x1.shape[2]
    est_sig_inv=np.zeros((x1.shape[0],3*x1.shape[2]))
    # note for model 2 and 3 you dont get a noise estimate
    for ind in np.arange(x1.shape[0]):
        for comp in range(3):
            if model=='v1':
                # apply masks to noisy input signal and inverse transform 
                _,est_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft(test_predictions[ind,:,:,comp]*(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
            elif model=='v2':
                if not(norm_input):
                    _,est_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft((test_predictions[ind,:,:,comp*2]+test_predictions[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
                else:
                    _,est_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft((test_predictions[ind,:,:,comp*2]+test_predictions[ind,:,:,comp*2+1]*1j)*np.max(np.abs(x1[ind,:,:,2*comp]+x1[ind,:,:,2*comp+1]*1j)), fs=sr, nperseg=nperseg, noverlap=noverlap)
            elif model=='v3':
                '''
                sig_noise_stft=x1[ind,:,:,2*comp]+x1[ind,:,:,2*comp+1]*1j
                A=np.power(np.e,y[ind,:,:,comp*3])-eps
                costheta=y[ind,:,:,comp*3+1]
                sintheta=y[ind,:,:,comp*3+2]
                _,tru_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft(sig_noise_stft*A*(costheta+sintheta*1j), 
                                                 fs=sr, nperseg=nperseg, noverlap=noverlap)
                A=np.power(np.e,test_predictions[ind,:,:,comp*3])-eps
                costheta=test_predictions[ind,:,:,comp*3+1]
                sintheta=test_predictions[ind,:,:,comp*3+2]
                _,est_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft(sig_noise_stft*A*(costheta+sintheta*1j), 
                                                 fs=sr, nperseg=nperseg, noverlap=noverlap) '''
                sig_noise_stft=x1[ind,:,:,2*comp]+x1[ind,:,:,2*comp+1]*1j
                _,est_sig_inv[ind,comp*nlen:(comp+1)*nlen]=signal.istft(sig_noise_stft*(np.power(np.e,test_predictions[ind,:,:,comp*3])-eps)*(test_predictions[ind,:,:,comp*3+1]+test_predictions[ind,:,:,comp*3+2]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
    return est_sig_inv

def comp_CC(signal, noisy_signal, denoised_signal):
    '''
    Compute the  normalized cross correlation betweeen the signal and denoised signal
    and the difference in normalized CC before and after denoising.

    Parameters
    ----------
    signal : float
        true signal
    noisy_signal : float
        true signal + noise
    denoised_signal : float
        denoised signal

    Returns
    -------
    CC, dCC : float
        cross correlation and difference in cross correlation

    '''
    CC1=correlate_template(signal, noisy_signal)
    CC2=correlate_template(signal, denoised_signal)
    return CC1, CC2

def comp_euclidean(signal, noisy_signal, denoised_signal):
    '''
    Compute the euclidean distance betweeen the signal and denoised signal
    and the difference in euclidean distance before and after denoising.

    Parameters
    ----------
    signal : float
        true signal
    noisy_signal : float
        true signal + noise
    denoised_signal : float
        denoised signal

    Returns
    -------
    EUC, dEUC : float
        euclidean distance and difference in euclidean distance

    '''
    EUC1=np.sqrt(np.sum((signal-noisy_signal) ** 2))
    EUC2=np.sqrt(np.sum((signal-denoised_signal) ** 2))
    return EUC1, EUC2

def comp_SNR(signal, noisy_signal, denoised_signal):
    try:
        ind=np.where(np.abs(signal)>=0.0005)[0][0]
        print(ind)
        if ind > 9 and ind < 118:
            SNR1=np.max(np.abs(signal[ind:]))/(2*np.std(noisy_signal[:ind]))
            SNR2=np.max(np.abs(denoised_signal[ind:]))/(2*np.std(denoised_signal[:ind]))
        else:
            SNR1=SNR2=np.nan
    except:
        SNR1=SNR2=np.nan
    return SNR1, SNR2

def comp_PGD(c1,c2,c3):
    return np.max(np.sqrt(c1**2+c2**2+c3**2))

def make_small_unet_v1(drop=0, ncomp=1, fac=1):
    
    if ncomp==1:
        input_layer=tf.keras.layers.Input(shape=(16,128,2)) # 1 Channel seismic data
    elif ncomp==3:
        input_layer=tf.keras.layers.Input(shape=(16,128,6)) # 3 Channel seismic data        
    
    # 1st level -- blue arrow
    level1=tf.keras.layers.Conv2D(8*fac,3,activation='relu',padding='same')(input_layer) # N filters, kernel Size, Strides, padding
    level1=tf.keras.layers.BatchNormalization()(level1)
    network=level1
    
    # 1st level -- blue arrow
    level1b=tf.keras.layers.Conv2D(8*fac,3,activation='relu',padding='same')(level1) # N filters, kernel Size, Strides, padding
    level1b=tf.keras.layers.BatchNormalization()(level1b)
    level1b=tf.keras.layers.Dropout(drop)(level1b)
    network=level1b
    
    # 1st --> 2nd level -- orange arrow
    level2=tf.keras.layers.Conv2D(8*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level2=tf.keras.layers.BatchNormalization()(level2)
    
    # 2nd level -- blue arrow
    level2b=tf.keras.layers.Conv2D(16*fac,3,activation='relu',padding='same')(level2) # N filters, kernel Size, Strides, padding
    level2b=tf.keras.layers.BatchNormalization()(level2b)
    level2b=tf.keras.layers.Dropout(drop)(level2b)
    network=level2b
    
    # 2nd --> 3rd level -- orange arrow
    level3=tf.keras.layers.Conv2D(16*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level3=tf.keras.layers.BatchNormalization()(level3)
    
    # 3rd level -- blue arrow
    level3b=tf.keras.layers.Conv2D(32*fac,3,activation='relu',padding='same')(level3) # N filters, kernel Size, Strides, padding
    level3b=tf.keras.layers.BatchNormalization()(level3b)
    level3b=tf.keras.layers.Dropout(drop)(level3b)
    network=level3b
    
    # 3rd --> 4th level -- orange arrow
    level4=tf.keras.layers.Conv2D(32*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level4=tf.keras.layers.BatchNormalization()(level4)
    
    # 4th level -- blue arrow
    level4b=tf.keras.layers.Conv2D(64*fac,3,activation='relu',padding='same')(level4) # N filters, kernel Size, Strides, padding
    level4b=tf.keras.layers.BatchNormalization()(level4b)  
    level4b=tf.keras.layers.Dropout(drop)(level4b)
    network=level4b
    
    # 4th --> 5th level -- orange arrow
    level5=tf.keras.layers.Conv2D(64*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level5=tf.keras.layers.BatchNormalization()(level5)
    
    # 5th level -- blue arrow
    level5b=tf.keras.layers.Conv2D(128*fac,3,activation='relu',padding='same')(level5) # N filters, kernel Size, Strides, padding
    level5b=tf.keras.layers.BatchNormalization()(level5b)
    level5b=tf.keras.layers.Dropout(drop)(level5b)
    network=level5b
    
    # # #Base of Network
    
    # 5th level --> 4th level -- green arrow + skip connection
    level4u=tf.keras.layers.Conv2DTranspose(64*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level4u) 
    network=tf.keras.layers.Concatenate()([network,level4b])

    # 4th level -- blue arrow
    level4b=tf.keras.layers.Conv2D(64*fac,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level4b)     
    
    # 4th level --> 3rd level -- green arrow + skip connection
    level3u=tf.keras.layers.Conv2DTranspose(32*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level3u) 
    network=tf.keras.layers.Concatenate()([network,level3b])

    # 3rd level -- blue arrow
    level3b=tf.keras.layers.Conv2D(32*fac,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level3b)  
    
    # 3rd level --> 2nd level -- green arrow + skip connection
    level2u=tf.keras.layers.Conv2DTranspose(16*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level2u) 
    network=tf.keras.layers.Concatenate()([network,level2b])

    # 2nd level -- blue arrow
    level2b=tf.keras.layers.Conv2D(16*fac,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level2b) 
    
    # 2nd level --> 1st level -- green arrow + skip connection
    level1u=tf.keras.layers.Conv2DTranspose(8*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level1u) 
    network=tf.keras.layers.Concatenate()([network,level1b])

    # 1st level -- blue arrow
    level1b=tf.keras.layers.Conv2D(8*fac,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level1b) 
    
    # End of network
    if ncomp==1:
        output = tf.keras.layers.Conv2D(1,1,activation='sigmoid',padding='same')(network)# N filters, Filter Size, Stride, padding
    else:
        output = tf.keras.layers.Conv2D(3,1,activation='sigmoid',padding='same')(network)# N filters, Filter Size, Stride, padding    
    model=tf.keras.models.Model(input_layer,output)
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='mse',optimizer=opt,metrics=['mse'])
    
    model.summary()
    
    tf.keras.utils.plot_model(model, to_file='denoising_model_v1_plot.png', show_shapes=True, show_layer_names=True)
    
    return model

def make_small_unet_v2(drop=0, ncomp=1, fac=1):
    
    if ncomp==1:
        input_layer=tf.keras.layers.Input(shape=(16,128,2)) # 1 Channel seismic data
    elif ncomp==3:
        input_layer=tf.keras.layers.Input(shape=(16,128,6)) # 1 Channel seismic data        
    
    # 1st level -- blue arrow
    level1=tf.keras.layers.Conv2D(8*fac,3,activation='relu',padding='same')(input_layer) # N filters, kernel Size, Strides, padding
    level1=tf.keras.layers.BatchNormalization()(level1)
    network=level1
    
    # 1st level -- blue arrow
    level1b=tf.keras.layers.Conv2D(8*fac,3,activation='relu',padding='same')(level1) # N filters, kernel Size, Strides, padding
    level1b=tf.keras.layers.BatchNormalization()(level1b)
    level1b=tf.keras.layers.Dropout(drop)(level1b)
    network=level1b
    
    # 1st --> 2nd level -- orange arrow
    level2=tf.keras.layers.Conv2D(8*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level2=tf.keras.layers.BatchNormalization()(level2)
    
    # 2nd level -- blue arrow
    level2b=tf.keras.layers.Conv2D(16*fac,3,activation='relu',padding='same')(level2) # N filters, kernel Size, Strides, padding
    level2b=tf.keras.layers.BatchNormalization()(level2b)
    level2b=tf.keras.layers.Dropout(drop)(level2b)
    network=level2b
    
    # 2nd --> 3rd level -- orange arrow
    level3=tf.keras.layers.Conv2D(16*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level3=tf.keras.layers.BatchNormalization()(level3)
    
    # 3rd level -- blue arrow
    level3b=tf.keras.layers.Conv2D(32*fac,3,activation='relu',padding='same')(level3) # N filters, kernel Size, Strides, padding
    level3b=tf.keras.layers.BatchNormalization()(level3b)
    level3b=tf.keras.layers.Dropout(drop)(level3b)
    network=level3b
    
    # 3rd --> 4th level -- orange arrow
    level4=tf.keras.layers.Conv2D(32*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level4=tf.keras.layers.BatchNormalization()(level4)
    
    # 4th level -- blue arrow
    level4b=tf.keras.layers.Conv2D(64*fac,3,activation='relu',padding='same')(level4) # N filters, kernel Size, Strides, padding
    level4b=tf.keras.layers.BatchNormalization()(level4b)  
    level4b=tf.keras.layers.Dropout(drop)(level4b)
    network=level4b
    
    # 4th --> 5th level -- orange arrow
    level5=tf.keras.layers.Conv2D(64*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level5=tf.keras.layers.BatchNormalization()(level5)
    
    # 5th level -- blue arrow
    level5b=tf.keras.layers.Conv2D(128*fac,3,activation='relu',padding='same')(level5) # N filters, kernel Size, Strides, padding
    level5b=tf.keras.layers.BatchNormalization()(level5b)
    level5b=tf.keras.layers.Dropout(drop)(level5b)
    network=level5b
    
    # # #Base of Network
    
    # 5th level --> 4th level -- green arrow + skip connection
    level4u=tf.keras.layers.Conv2DTranspose(64*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level4u) 
    network=tf.keras.layers.Concatenate()([network,level4b])

    # 4th level -- blue arrow
    level4b=tf.keras.layers.Conv2D(64*fac,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level4b)     
    
    # 4th level --> 3rd level -- green arrow + skip connection
    level3u=tf.keras.layers.Conv2DTranspose(32*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level3u) 
    network=tf.keras.layers.Concatenate()([network,level3b])

    # 3rd level -- blue arrow
    level3b=tf.keras.layers.Conv2D(32*fac,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level3b)  
    
    # 3rd level --> 2nd level -- green arrow + skip connection
    level2u=tf.keras.layers.Conv2DTranspose(16*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level2u) 
    network=tf.keras.layers.Concatenate()([network,level2b])

    # 2nd level -- blue arrow
    level2b=tf.keras.layers.Conv2D(16*fac,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level2b) 
    
    # 2nd level --> 1st level -- green arrow + skip connection
    level1u=tf.keras.layers.Conv2DTranspose(8*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level1u) 
    network=tf.keras.layers.Concatenate()([network,level1b])

    # 1st level -- blue arrow
    level1b=tf.keras.layers.Conv2D(8*fac,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level1b) 
    
    # End of network
    if ncomp==1:
        output = tf.keras.layers.Conv2D(1,1,activation='linear',padding='same')(network)# N filters, Filter Size, Stride, padding
    elif ncomp==3:
        output = tf.keras.layers.Conv2D(6,1,activation='linear',padding='same')(network)# N filters, Filter Size, Stride, padding    
    model=tf.keras.models.Model(input_layer,output)
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='mse',optimizer=opt,metrics=['mse'])
    
    model.summary()
    
    tf.keras.utils.plot_model(model, to_file='denoising_model_v2_plot.png', show_shapes=True, show_layer_names=True)
    
    return model

def make_small_unet_v3(drop=0, ncomp=1, fac=1):
    if ncomp==1:
        input_layer=tf.keras.layers.Input(shape=(16,128,2)) # 1 Channel seismic data
    elif ncomp==3:
        input_layer=tf.keras.layers.Input(shape=(16,128,6)) # 1 Channel seismic data    
    
    # 1st level -- blue arrow
    level1=tf.keras.layers.Conv2D(8*fac,3,activation='relu',padding='same')(input_layer) # N filters, kernel Size, Strides, padding
    level1=tf.keras.layers.BatchNormalization()(level1)
    network=level1
    
    # 1st level -- blue arrow
    level1b=tf.keras.layers.Conv2D(8*fac,3,activation='relu',padding='same')(level1) # N filters, kernel Size, Strides, padding
    level1b=tf.keras.layers.BatchNormalization()(level1b)
    level1b=tf.keras.layers.Dropout(drop)(level1b)
    network=level1b
    
    # 1st --> 2nd level -- orange arrow
    level2=tf.keras.layers.Conv2D(8*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level2=tf.keras.layers.BatchNormalization()(level2)
    
    # 2nd level -- blue arrow
    level2b=tf.keras.layers.Conv2D(16*fac,3,activation='relu',padding='same')(level2) # N filters, kernel Size, Strides, padding
    level2b=tf.keras.layers.BatchNormalization()(level2b)
    level2b=tf.keras.layers.Dropout(drop)(level2b)
    network=level2b
    
    # 2nd --> 3rd level -- orange arrow
    level3=tf.keras.layers.Conv2D(16*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level3=tf.keras.layers.BatchNormalization()(level3)
    
    # 3rd level -- blue arrow
    level3b=tf.keras.layers.Conv2D(32*fac,3,activation='relu',padding='same')(level3) # N filters, kernel Size, Strides, padding
    level3b=tf.keras.layers.BatchNormalization()(level3b)
    level3b=tf.keras.layers.Dropout(drop)(level3b)
    network=level3b
    
    # 3rd --> 4th level -- orange arrow
    level4=tf.keras.layers.Conv2D(32*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level4=tf.keras.layers.BatchNormalization()(level4)
    
    # 4th level -- blue arrow
    level4b=tf.keras.layers.Conv2D(64*fac,3,activation='relu',padding='same')(level4) # N filters, kernel Size, Strides, padding
    level4b=tf.keras.layers.BatchNormalization()(level4b)  
    level4b=tf.keras.layers.Dropout(drop)(level4b)
    network=level4b
    
    # 4th --> 5th level -- orange arrow
    level5=tf.keras.layers.Conv2D(64*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    level5=tf.keras.layers.BatchNormalization()(level5)
    
    # 5th level -- blue arrow
    level5b=tf.keras.layers.Conv2D(128*fac,3,activation='relu',padding='same')(level5) # N filters, kernel Size, Strides, padding
    level5b=tf.keras.layers.BatchNormalization()(level5b)
    level5b=tf.keras.layers.Dropout(drop)(level5b)
    network=level5b
    
    # # #Base of Network
    
    # 5th level --> 4th level -- green arrow + skip connection
    level4u=tf.keras.layers.Conv2DTranspose(64*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level4u) 
    network=tf.keras.layers.Concatenate()([network,level4b])

    # 4th level -- blue arrow
    level4b=tf.keras.layers.Conv2D(64*fac,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level4b)     
    
    # 4th level --> 3rd level -- green arrow + skip connection
    level3u=tf.keras.layers.Conv2DTranspose(32*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level3u) 
    network=tf.keras.layers.Concatenate()([network,level3b])

    # 3rd level -- blue arrow
    level3b=tf.keras.layers.Conv2D(32*fac,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level3b)  
    
    # 3rd level --> 2nd level -- green arrow + skip connection
    level2u=tf.keras.layers.Conv2DTranspose(16*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level2u) 
    network=tf.keras.layers.Concatenate()([network,level2b])

    # 2nd level -- blue arrow
    level2b=tf.keras.layers.Conv2D(16*fac,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level2b) 
    
    # 2nd level --> 1st level -- green arrow + skip connection
    level1u=tf.keras.layers.Conv2DTranspose(8*fac,3,strides=2,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
    network=tf.keras.layers.BatchNormalization()(level1u) 
    network=tf.keras.layers.Concatenate()([network,level1b])

    # 1st level -- blue arrow
    level1b=tf.keras.layers.Conv2D(8*fac,3,activation='relu',padding='same')(network) # N filters, kernel Size, Strides, padding
    network=tf.keras.layers.BatchNormalization()(level1b) 
    
    # End of network
    # this is the logarithm of the amplitude ratio
    output = tf.keras.layers.Conv2D(9,1,activation='linear',padding='same')(network)        
    model=tf.keras.models.Model(input_layer,output)
    
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(loss=['mse'],optimizer=opt,metrics=['mse'])
    
    model.summary()
    
    tf.keras.utils.plot_model(model, to_file='denoising_model_v3_plot.png', show_shapes=True, show_layer_names=True)
    
    return model

def plot_training_curves(model_save_file):
    training_stats = pd.read_csv("./"+model_save_file+'.csv', delimiter=',')
    fig, (ax1) = plt.subplots(1, 1, figsize=(8,5))
    ax1.plot(training_stats['epoch'].values,training_stats['mean_squared_error'].values,label="MSE")
    ax1.plot(training_stats['epoch'].values,training_stats['val_mean_squared_error'].values,label="val MSE")
    ax1.legend()
    ax1.set_xlabel('Epoch')
    ax1.set_title(model_save_file)
    return None

def stft_plot(trace, sr, nperseg, noverlap, plot=False):
    
    '''
    
    Plots STFT and makes time and frequency vectors
    
    '''
    f, t, Zxx = signal.stft(trace, fs=sr, nperseg=nperseg, noverlap=noverlap)

    if plot:
        fig,ax=plt.subplots(nrows=2,ncols=1)
        # for istft need (x.shape[axis] - nperseg) % (nperseg-noverlap) == 0)
        ax[0].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        ax[0].set_title('STFT Magnitude')
        ax[0].set_ylabel('Frequency [Hz]')
        ax[0].set_xlabel('Time [sec]')
        print("len(f) is "+str(len(f)))
        print("len(t) is "+str(len(t)))
    
        _,Zxx_inv=signal.istft(Zxx,fs=sr,nperseg=31, noverlap=30)
        ax[1].plot(trace,label='original')
        ax[1].plot(Zxx_inv,'--',label='from inv')
        ax[1].legend()
    
    return t, f
    
def plot_generator_results(model,t,f,x,y,sigs,noise,x1,nlen,sr,nperseg,noverlap,norm_input,eps=1e-9):
    """
    Plots the result of the generator 

    Parameters
    ----------
    t : float array
        time vector of STFT
    f : float array
        frequency vector of STFT
    x : complex float array 
        model input (STFT(signal+noise))
    y : real or complex float array
        model output
    sigs : float array
        each row is a signal trace
    noise : float array
        each row is a noise trace
    x1 : TYPE
        DESCRIPTION.
    nlen : int
        length of signal
    sr : int
        sample rate
    nperseg : int
        number per segment input into STFT
    noverlap : int
        noverlap input into STFT

    Returns
    -------
    None.

    """
    
    minrange,maxrange=0,2
    for count, ind in enumerate(range(minrange,maxrange)):
        fig, axs = plt.subplots(nrows=4,ncols=6,figsize=(20,15),sharex=True)
        # plot original data and noise
        for comp in range(3):
            axs[0,comp*2].plot(t,sigs[ind,comp*nlen:(comp+1)*nlen], label='signal')
            axs[0,comp*2].plot(t,sigs[ind,comp*nlen:(comp+1)*nlen]+noise[ind,comp*nlen:(comp+1)*nlen], color=(0.6,0.6,0.6), alpha=0.5, label='signal+noise')
            axs[0,comp*2].legend()
            if comp==0:
                axs[0,comp*2].set_title('North')
            if comp==1:
                axs[0,comp*2].set_title('East')
            if comp==2:
                axs[0,comp*2].set_title('Vertical')
            axs[0,comp*2+1].plot(t,noise[ind,comp*nlen:(comp+1)*nlen], label='noise')
            axs[0,comp*2+1].plot(t,sigs[ind,comp*nlen:(comp+1)*nlen]+noise[ind,comp*nlen:(comp+1)*nlen], color=(0.6,0.6,0.6), alpha=0.5, label='signal+noise')
            axs[0,comp*2+1].legend()
            axs[0,comp*2+1].set_title('Original noise')   
            lim=np.max(np.abs(sigs[ind,comp*nlen:(comp+1)*nlen]+noise[ind,comp*nlen:(comp+1)*nlen]))
            axs[0,comp*2].set_ylim((-lim,lim))
            axs[0,comp*2+1].set_ylim((-lim,lim))
            # plot the real and imaginary parts of the stft(signal+noise)
            axs[1,comp*2].pcolormesh(t, f, np.abs(x[ind,:,:,comp*2]), shading='gouraud')
            axs[1,comp*2].set_title('Re(STFT(signal+noise))')
            axs[1,comp*2+1].pcolormesh(t, f, np.abs(x[ind,:,:,comp*2+1]), shading='gouraud')
            axs[1,comp*2+1].set_title('Im(STFT(signal+noise))')
            # plot the output signal and noise masks
            axs[2,comp*2].pcolormesh(t, f, y[ind,:,:,comp], shading='gouraud')
            axs[2,comp*2].set_title('Signal mask (true)')
            axs[2,comp*2+1].pcolormesh(t, f, 1-y[ind,:,:,comp], shading='gouraud')  
            axs[2,comp*2+1].set_title('Noise mask (true)')
            # apply masks to noisy input signal and inverse transform 
            true_signal=sigs[ind,comp*nlen:(comp+1)*nlen]
            true_noise=noise[ind,comp*nlen:(comp+1)*nlen]
            if model=='v1':
                # apply masks to noisy input signal and inverse transform 
                _,tru_sig_inv=signal.istft(y[ind,:,:,comp]*(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)
                _,tru_noise_inv=signal.istft((1-y[ind,:,:,comp])*(x1[ind,:,:,comp*2]+x1[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)    
            elif model=='v2':
                if not(norm_input):
                    _,tru_sig_inv=signal.istft((y[ind,:,:,comp*2]+y[ind,:,:,comp*2+1]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)  
                else:
                    _,tru_sig_inv=signal.istft(((y[ind,:,:,comp*2]+y[ind,:,:,comp*2+1]*1j)*np.max(np.abs(x1[ind,:,:,2*comp]+x1[ind,:,:,2*comp+1]*1j))), fs=sr, nperseg=nperseg, noverlap=noverlap)  
                tru_noise_inv=true_signal+true_noise-tru_sig_inv
            if model=='v3':
                sig_noise_stft=x1[ind,:,:,2*comp]+x1[ind,:,:,2*comp+1]*1j
                _,tru_sig_inv=signal.istft(sig_noise_stft*(np.power(np.e,y[ind,:,:,comp*3])-eps)*(y[ind,:,:,comp*3+1]+y[ind,:,:,comp*3+2]*1j), fs=sr, nperseg=nperseg, noverlap=noverlap)  
                tru_noise_inv=true_signal+true_noise-tru_sig_inv
            axs[3,comp*2].plot(t,true_signal, label='true signal')
            axs[3,comp*2].plot(t, tru_sig_inv, alpha=0.75, color=(0.6,0,0), label='reconstructed signal')
            axs[3,comp*2].legend()
            axs[3,comp*2].set_title('Denoised signal')
            axs[3,comp*2].set_ylim((-lim,lim))
            axs[3,comp*2+1].plot(t,true_noise, label='true noise')
            axs[3,comp*2+1].plot(t, tru_noise_inv, alpha=0.75, color=(0.6,0,0), label='reconstructed noise')
            axs[3,comp*2+1].set_title('Designaled noise')
            axs[3,comp*2+1].set_ylim((-lim,lim))
            axs[3,comp*2+1].legend()
        return None

def plot_data(x_data,n_data):
    '''
    Plot signal and noise data

    Parameters
    ----------
    x_data : float array
        signal data
    n_data : float array
        noise data

    Returns
    -------
    None.

    '''
    # plot ps to check
    fig, ax=plt.subplots()
    for ii in range(40): #x_data.shape[0]):
        plt.plot(x_data[ii,:]/np.max(np.abs(x_data[ii,:]))+ii)
    plt.title('signals')
    ax.axvline(x=256//2)
    ax.axvline(x=256+256//2)
    ax.axvline(x=2*256+256//2)
        
    # plot noise to check
    plt.figure()
    for ii in range(40): #n_data.shape[0]):
        plt.plot(n_data[ii,:])
    plt.title('noise')
    
    # plot pgd to check
    pgd=np.zeros(270000)
    for ii in range(len(pgd)):
        n=x_data[ii,:256]
        e=x_data[ii,256:2*256]
        z=x_data[ii,2*256:]
        pgd[ii]=np.max(np.sqrt(n**2+e**2+z**2))
    plt.figure()
    plt.hist(pgd,bins=100)
    plt.xlabel('PGD')
    plt.ylabel('Frequency')
    return None
    
def plot_denoised_signals(t, f, sigs, noise, maxrange, nlen, x, y, test_predictions, CC, SNR, tru_sig_inv, tru_noise_inv, est_sig_inv, est_noise_inv, mlim=100, minrange=0):
    '''
    Plot true and denoised signals.

    Parameters
    ----------
    t : float array
        time vector of STFT
    f : float array
        frequency vector of STFT
    sigs : float array
        each row is a signal trace
    noise : float array
        each row is a noise trace
    maxrange : TYPE
        DESCRIPTION.
    nlen : TYPE
        DESCRIPTION.
    x : complex float array 
        model input (STFT(signal+noise))
    y : real or complex float array
        model output
    test_predictions : complex array
        model predictions
    CC : float array
        cross correlations
    SNR : float array
        SNRs
    tru_sig_inv : float array
        inverse transformed true signal 
    tru_noise_inv : float array
        inverse transformed true noise
    est_sig_inv : float array
        inverse transformed estimated signal
    est_noise_inv : float array
        inverse transformed estimated noise

    Returns
    -------
    None.

    '''
    
    for ind in range(minrange,maxrange):
        if SNR[ind,2] < 0.5: #(np.max(np.abs(sigs[ind,:])) < mlim):
            thisplot=1
            print(str(ind))
        else:
            thisplot=0
        if thisplot:
            fig, axs = plt.subplots(nrows=5,ncols=6,figsize=(20,15),sharex=True,num=ind)
        # plot original data and noise
        lim=np.max(np.abs(sigs[ind,:]+noise[ind,:]))*1.1
        if thisplot:
            for comp in range(3):
                axs[0,comp*2].plot(t,sigs[ind,comp*nlen:(comp+1)*nlen], label='signal')
                axs[0,comp*2].plot(t,sigs[ind,comp*nlen:(comp+1)*nlen]+noise[ind,comp*nlen:(comp+1)*nlen], color=(0.6,0.6,0.6), alpha=0.5, label='signal+noise')
                if comp==0:
                    axs[0,comp*2].legend()
                axs[0,comp*2].set_title('Original signal')
                axs[0,comp*2+1].plot(t,noise[ind,comp*nlen:(comp+1)*nlen], label='noise')
                axs[0,comp*2+1].plot(t,sigs[ind,comp*nlen:(comp+1)*nlen]+noise[ind,comp*nlen:(comp+1)*nlen], color=(0.6,0.6,0.6), alpha=0.5, label='signal+noise')
                # axs[0,comp*2+1].legend()
                axs[0,comp*2+1].set_title('Original noise')   
                
                #axs[0,comp*2].set_ylim((-lim,lim))
                #axs[0,comp*2+1].set_ylim((-lim,lim))
                # plot the real and imaginary parts of the stft(signal+noise)
                axs[1,comp*2].pcolormesh(t, f, np.abs(x[ind,:,:,comp*2]), shading='gouraud')
                axs[1,comp*2].set_title('Re(STFT(signal+noise))')
                axs[1,comp*2+1].pcolormesh(t, f, np.abs(x[ind,:,:,comp*2+1]), shading='gouraud')
                axs[1,comp*2+1].set_title('Im(STFT(signal+noise))')
                
                # plot the output signal and noise masks
                axs[2,comp*2].pcolormesh(t, f, y[ind,:,:,comp], shading='gouraud', vmin=0, vmax=1)
                axs[2,comp*2].set_title('Signal mask (true)')
                axs[2,comp*2+1].pcolormesh(t, f, 1-y[ind,:,:,comp], shading='gouraud', vmin=0, vmax=1)
                axs[2,comp*2+1].set_title('Noise mask (true)')
                
                #noise mask predicted
                axs[3,comp*2].pcolormesh(t, f, test_predictions[ind,:,:,comp], shading='gouraud', vmin=0, vmax=1)
                axs[3,comp*2].set_title('Signal mask (predicted)')
                axs[3,comp*2+1].pcolormesh(t, f, 1-test_predictions[ind,:,:,comp], shading='gouraud', vmin=0, vmax=1)
                axs[3,comp*2+1].set_title('Noise mask (predcited)')
            
                # plot predicted signals
                axs[4,comp*2].plot(t, sigs[ind,comp*nlen:(comp+1)*nlen], color=(0,0,0.6), label='true')
                axs[4,comp*2].plot(t, tru_sig_inv[ind,comp*nlen:(comp+1)*nlen], alpha=0.75, color=(0.6,0,0), label='reconstructed')
                axs[4,comp*2].plot(t, est_sig_inv[ind,comp*nlen:(comp+1)*nlen], alpha=0.75, color=(0,0.6,0), label='ML')
                if comp==0:
                    axs[4,comp*2].legend()
                axs[4,comp*2].set_title('Denoised signal - CC='+str(np.round(CC[ind,comp]*100)/100)+" SNR="+str(np.round(SNR[ind,comp]*100)/100))
                #axs[4,comp*2].set_ylim((-lim,lim))
                axs[4,comp*2+1].plot(t, noise[ind,comp*nlen:(comp+1)*nlen], color=(0,0,0.6), label='true noise')
                axs[4,comp*2+1].plot(t, tru_noise_inv[ind,comp*nlen:(comp+1)*nlen], alpha=0.75, color=(0.6,0,0), label='reconstructed noise')
                axs[4,comp*2+1].plot(t, est_noise_inv[ind,comp*nlen:(comp+1)*nlen], alpha=0.75, color=(0,0.6,0), label='ML noise')
                axs[4,comp*2+1].set_title('Designaled noise')
                #axs[4,comp*2+1].set_ylim((-lim,lim))
                    #axs[3,comp*2+1].legend()
            for ii in [0,4]:
                for jj in range(6):
                    axs[ii,jj].set_ylim((-lim,lim))
    return None

def plot_denoising_comparison(t, f, sigs, noise, maxrange, nlen, x, y, test_predictions, CC, SNR, tru_sig_inv, tru_noise_inv, est_sig_inv, est_noise_inv, mlim=100, minrange=0):
    '''
    Plot true and denoised signals.

    Parameters
    ----------
    t : float array
        time vector of STFT
    f : float array
        frequency vector of STFT
    sigs : float array
        each row is a signal trace
    noise : float array
        each row is a noise trace
    maxrange : TYPE
        DESCRIPTION.
    nlen : TYPE
        DESCRIPTION.
    x : complex float array 
        model input (STFT(signal+noise))
    y : real or complex float array
        model output
    test_predictions : complex array
        model predictions
    CC : float array
        cross correlations
    SNR : float array
        SNRs
    tru_sig_inv : float array
        inverse transformed true signal 
    tru_noise_inv : float array
        inverse transformed true noise
    est_sig_inv : float array
        inverse transformed estimated signal
    est_noise_inv : float array
        inverse transformed estimated noise

    Returns
    -------
    None.

    '''
    
    for ind in range(minrange,maxrange):
        if SNR[ind,2] < 0.5: #(np.max(np.abs(sigs[ind,:])) < mlim):
            thisplot=1
            print(str(ind))
        else:
            thisplot=0
        if thisplot:
            fig, axs = plt.subplots(nrows=5,ncols=6,figsize=(20,15),sharex=True,num=ind)
        # plot original data and noise
        lim=np.max(np.abs(sigs[ind,:]+noise[ind,:]))*1.1
        if thisplot:
            for comp in range(3):
                axs[0,comp*2].plot(t,sigs[ind,comp*nlen:(comp+1)*nlen], label='signal')
                axs[0,comp*2].plot(t,sigs[ind,comp*nlen:(comp+1)*nlen]+noise[ind,comp*nlen:(comp+1)*nlen], color=(0.6,0.6,0.6), alpha=0.5, label='signal+noise')
                if comp==0:
                    axs[0,comp*2].legend()
                axs[0,comp*2].set_title('Original signal')
                axs[0,comp*2+1].plot(t,noise[ind,comp*nlen:(comp+1)*nlen], label='noise')
                axs[0,comp*2+1].plot(t,sigs[ind,comp*nlen:(comp+1)*nlen]+noise[ind,comp*nlen:(comp+1)*nlen], color=(0.6,0.6,0.6), alpha=0.5, label='signal+noise')
                # axs[0,comp*2+1].legend()
                axs[0,comp*2+1].set_title('Original noise')   
                
                #axs[0,comp*2].set_ylim((-lim,lim))
                #axs[0,comp*2+1].set_ylim((-lim,lim))
                # plot the real and imaginary parts of the stft(signal+noise)
                axs[1,comp*2].pcolormesh(t, f, np.abs(x[ind,:,:,comp*2]), shading='gouraud')
                axs[1,comp*2].set_title('Re(STFT(signal+noise))')
                axs[1,comp*2+1].pcolormesh(t, f, np.abs(x[ind,:,:,comp*2+1]), shading='gouraud')
                axs[1,comp*2+1].set_title('Im(STFT(signal+noise))')
                
                # plot the output signal and noise masks
                axs[2,comp*2].pcolormesh(t, f, y[ind,:,:,comp], shading='gouraud', vmin=0, vmax=1)
                axs[2,comp*2].set_title('Signal mask (true)')
                axs[2,comp*2+1].pcolormesh(t, f, 1-y[ind,:,:,comp], shading='gouraud', vmin=0, vmax=1)
                axs[2,comp*2+1].set_title('Noise mask (true)')

            
                # plot predicted signals
                axs[4,comp*2].plot(t, sigs[ind,comp*nlen:(comp+1)*nlen], color=(0,0,0.6), label='true')
                axs[4,comp*2].plot(t, tru_sig_inv[ind,comp*nlen:(comp+1)*nlen], alpha=0.75, color=(0.6,0,0), label='reconstructed')
                axs[4,comp*2].plot(t, est_sig_inv[ind,comp*nlen:(comp+1)*nlen], alpha=0.75, color=(0,0.6,0), label='ML')
                if comp==0:
                    axs[4,comp*2].legend()
                axs[4,comp*2].set_title('Denoised signal - CC='+str(np.round(CC[ind,comp]*100)/100)+" SNR="+str(np.round(SNR[ind,comp]*100)/100))
                #axs[4,comp*2].set_ylim((-lim,lim))
                axs[4,comp*2+1].plot(t, noise[ind,comp*nlen:(comp+1)*nlen], color=(0,0,0.6), label='true noise')
                axs[4,comp*2+1].plot(t, tru_noise_inv[ind,comp*nlen:(comp+1)*nlen], alpha=0.75, color=(0.6,0,0), label='reconstructed noise')
                axs[4,comp*2+1].plot(t, est_noise_inv[ind,comp*nlen:(comp+1)*nlen], alpha=0.75, color=(0,0.6,0), label='ML noise')
                axs[4,comp*2+1].set_title('Designaled noise')
                #axs[4,comp*2+1].set_ylim((-lim,lim))
                    #axs[3,comp*2+1].legend()
            for ii in [0,4]:
                for jj in range(6):
                    axs[ii,jj].set_ylim((-lim,lim))
    return None

def plot_and_save_performance(model_save_file, SNR, dSNR, CC, dCC, EUC, dEUC, truePGD, noisyPGD, predictedPGD):
    '''
    
    Plots signal-to-noise ratio, cross correlation, and variance 
    reduction metrics for each model and saves them.

    Parameters
    ----------
    model_save_file : str
        name of model file that will correspond to name of saved metrics file
    SNR : float
        signal to noise ratio from comp_SNR
    CC : float
        cross correlation between true and denoised signals from comp_CC
    VR : float
        variance reduction from comp_VR (not really used, unstable)
    AMP: float

    Returns
    -------
    None.
    
    '''
    plt.figure()
    plt.plot(SNR[:,0],CC[:,0],'o',label='east')
    plt.plot(SNR[:,1],CC[:,1],'o',label='north')
    plt.plot(SNR[:,2],CC[:,2],'o',label='vertical')
    plt.xlabel('SNR')
    plt.ylabel('CC')
    plt.ylim((-1,1))
    plt.xlim((0,10))
    plt.legend()
    snrbins=np.arange(0.25,10,0.5)
    medsnrs=np.zeros(len(snrbins))
    for ii, snrbin in enumerate(snrbins):
        medsnrs[ii]=np.median(CC[np.where((SNR>=snrbin-0.25) & (SNR<snrbin+0.25))])
    plt.plot(snrbins,medsnrs,'ko')
    plt.title(model_save_file)
    
    plt.figure()
    plt.plot(SNR[:,0],EUC[:,0],'o',label='east')
    plt.plot(SNR[:,1],EUC[:,1],'o',label='north')
    plt.plot(SNR[:,2],EUC[:,2],'o',label='vertical')
    plt.xlabel('SNR')
    plt.ylabel('VR')
    plt.ylim((0,100))
    plt.xlim((0,10))
    plt.legend()
    
    pickle.dump([SNR, dSNR, CC, dCC, EUC, dEUC, truePGD, noisyPGD, predictedPGD], open(model_save_file[:-3]+'.res', "wb"))
    return None

def main():
    make_small_unet_v1(ncomp=3, fac=1)

if __name__ == "__main__":
    main()
