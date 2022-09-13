#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:29:11 2021

MODEL 1: Train a CNN to denoise 3 component GNSS data by predicting a real-valued mask

@author: amt
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gnss_tools
import h5py
import datetime
import pickle

# SET OPTIONS
train=False # # do you want to train?
plots=True # # do you want to make some plots?
epos=50 # how many epocs?
fac=4 # model size
sr=1 # sample rate
eps=1e-9 # epsilon
drop=0.2 # model drop rate
nlen=128 # window length
nperseg=31 # param for STFT
noverlap=30 # param for STFT
norm_input=True # leave this alone

# LOAD THE DATA
print("LOADING DATA")
x_data = h5py.File('nd3_data.hdf5','r')
n_data = h5py.File('729k_noise.hdf5','r')
sl_data = h5py.File('softlayers.hdf5','r')
x_data = x_data['nd3_data'][:,:]
n_data = n_data['729k_noise'][:,:]
sl_data = sl_data['layers'][:,:]
x_data = np.concatenate((x_data,sl_data))
np.random.seed(0)
np.random.shuffle(x_data)

# CALCULATE PGD
pgd=np.zeros(x_data.shape[0])
for ii in range(x_data.shape[0]):
    pgd[ii]=np.max(np.sqrt((x_data[ii,:256])**2+(x_data[ii,256:512])**2+(x_data[ii,512:])**2))

# CULL BASED ON PGD
inds=np.empty(0)
for ii in np.arange(0,15,0.1):
    ind=np.where((pgd>ii) & (pgd<=ii+0.1))[0]
    if ii < 1:
        inds=np.append(inds,ind[:60000])
    else:
        inds=np.append(inds,ind)
inds=inds.astype(dtype=int)        
x_data=x_data[inds,:]       

# SET MODEL FILE NAME
if train:
    if not(norm_input):
        model_save_file="quickie_3comp_v1_"+str(fac)+"_"+str(datetime.datetime.today().month)+"-"+str(datetime.datetime.today().day)+".tf"
    else:
        model_save_file="quickie_3comp_norm_input_v1_"+str(fac)+"_"+str(datetime.datetime.today().month)+"-"+str(datetime.datetime.today().day)+".tf"
else:
    if not(norm_input):
        model_save_file="quickie_3comp_v1_"+str(fac)+"_8-18.tf"  
    else:
        model_save_file="quickie_3comp_norm_input_v1_"+str(fac)+"_8-18.tf"     

# MAKE TRAINING AND TESTING DATA
print("MAKE TRAINING AND TESTING DATA")
siginds=np.arange(x_data.shape[0])
np.random.shuffle(siginds)
x_train_inds=np.sort(siginds[:int(0.9*len(siginds))])
x_test_inds=np.sort(siginds[int(0.9*len(siginds)):])
noiseinds=np.arange(n_data.shape[0])
np.random.shuffle(noiseinds)
n_train_inds=np.sort(noiseinds[:int(0.9*len(noiseinds))])
n_test_inds=np.sort(noiseinds[int(0.9*len(noiseinds)):])

# MAKE STFT TIME AND FREQUENCY VECTORS
t,f=gnss_tools.stft_plot(x_data[0,:128], sr, nperseg, noverlap)

# DATA GENERATOR
print("FIRST PASS WITH DATA GENERATOR")
my_data=gnss_tools.stft_3comp_data_generator('v1',32,x_data[x_train_inds,:],n_data[n_train_inds,:],sr,nperseg,noverlap,norm_input)
x,y=next(my_data)
my_test_data=gnss_tools.stft_3comp_data_generator('v1',50,x_data[x_train_inds,:],n_data[n_train_inds,:],sr,nperseg,noverlap,norm_input,valid=True)
x,y,sigs,noise,x1=next(my_test_data)

# # PLOT GENERATOR RESULTS
# print("PLOT GENERATOR RESULTS")
# gnss_tools.plot_generator_results('v1',t,f,x,y,sigs,noise,x1,nlen,sr,nperseg,noverlap,norm_input)
        
# BUILD THE MODEL
print("BUILD THE MODEL")
model=gnss_tools.make_small_unet_v1(drop=drop,ncomp=3,fac=fac)

'''
        
# ADD SOME CHECKPOINTS
print("ADDING CHECKPOINTS")
checkpoint_filepath = './checks/'+model_save_file+'_{epoch:04d}.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_weights_only=True, verbose=1,
    monitor='val_acc', mode='max', save_best_only=True)

# TRAIN THE MODEL
print("TRAINING!!!")
if train:
    batch_size=32
    # if resume:
    #     print('Resuming training results from '+model_save_file)
    #     model.load_weights(checkpoint_filepath)
    # else:
    print('Training model and saving results to '+model_save_file)
        
    csv_logger = tf.keras.callbacks.CSVLogger(model_save_file+".csv", append=True)
    history=model.fit_generator(gnss_tools.stft_3comp_data_generator('v1',batch_size,x_data[x_train_inds,:],n_data[n_train_inds,:],sr,nperseg, noverlap, norm_input),
                        steps_per_epoch=(len(x_train_inds))//batch_size,
                        validation_data=gnss_tools.stft_3comp_data_generator('v1',batch_size,x_data[x_test_inds,:],n_data[n_test_inds,:],sr,nperseg, noverlap, norm_input),
                        validation_steps=(len(x_test_inds))//batch_size,
                        epochs=epos, callbacks=[model_checkpoint_callback,csv_logger])
    model.save_weights("./"+model_save_file)
else:
    print('Loading training results from '+model_save_file)
    model.load_weights("./"+model_save_file)

# PLOT TRAINING STATS
print("PLOTTING TRAINING STATS")
gnss_tools.plot_training_curves(model_save_file)

# MAKE SOME PREDICTIONS
print("PREDICTING")
maxrange=x_test_inds.shape[0]
my_test_data=gnss_tools.stft_3comp_data_generator('v1',maxrange,x_data[x_test_inds,:],n_data[n_test_inds,:],sr,nperseg,noverlap,norm_input,valid=True)
x,y,sigs,noise,x1=next(my_test_data)
test_predictions=model.predict(x)

# GET DENOISED SIGNALS FROM OUTPUT
print("GETTING DENOISED SIGNALS FROM MODEL OUTPUT")
tru_sig_inv, tru_noise_inv, est_sig_inv, est_noise_inv=gnss_tools.output_2_data('v1',y,x1,test_predictions,sr,nperseg,noverlap,norm_input)

# SAVE SUBSET FOR PLOTTING PURPOSES
goto=maxrange
with open('model1_v'+str(fac)+'_results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([x[:goto,:,:,:],y[:goto,:,:,:],sigs[:goto,:],noise[:goto,:],x1[:goto,:,:,:],test_predictions[:goto,:,:,:]], f)

# # CALCULATE SOME METRICS
# # VR=np.zeros((int(maxrange),3))
# SNR=np.zeros((int(maxrange),3))
# CC=np.zeros((int(maxrange),3))
# for ii in range(maxrange):
#     for comp in range(3):
#         true_noise=noise[ii,comp*nlen:(comp+1)*nlen]
#         true_signal=sigs[ii,comp*nlen:(comp+1)*nlen]
#         #inds=np.where(np.abs(true_signal)>0.00001)[0]
#         #VR[ii,comp]=gnss_tools.comp_VR(true_signal,est_sig_inv[ii,comp*nlen:(comp+1)*nlen])    
#         _, CC[ii,comp]=gnss_tools.comp_CC(true_signal,true_noise+true_signal,est_sig_inv[ii,comp*nlen:(comp+1)*nlen])      
#         SNR[ii,comp]=gnss_tools.comp_SNR(true_signal, true_noise)

# # PLOT SOME EXAMPLES        
# gnss_tools.plot_denoised_signals(t, f, sigs, noise, maxrange, 
#                                   nlen, x, y, test_predictions, CC, SNR, tru_sig_inv, 
#                                   tru_noise_inv, est_sig_inv, est_noise_inv,minrange=0)
'''
