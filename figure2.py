#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 20:32:36 2022

Plot training data

@author: amt
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import numpy as np
import matplotlib.pylab as pl

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
        print(len(inds))
    else:
        inds=np.append(inds,ind)
inds=inds.astype(dtype=int)        
x_data=x_data[inds,:]         

# RECALCULATE PGD
pgd=np.zeros(x_data.shape[0])
for ii in range(x_data.shape[0]):
    pgd[ii]=np.max(np.sqrt((x_data[ii,:256])**2+(x_data[ii,256:512])**2+(x_data[ii,512:])**2))


for tmp in range(78,79):
    inds=[]
    for ii in np.arange(0,1.5,0.1):
        ind=np.where((pgd>ii) & (pgd<ii+0.5))[0][tmp]
        inds=np.append(inds,ind)
    inds=inds.astype(dtype=int)
    
    #%% Plot cell
    t=np.arange(46)
    fig = plt.figure(tight_layout=True, figsize=(9,9), num=tmp)
    gs = gridspec.GridSpec(5, 3)
    ax0 = fig.add_subplot(gs[:3,0])
    ax1= fig.add_subplot(gs[:3,1], sharey=ax0)
    #ax1.axes.get_yaxis().set_ticks([])
    ax2 = fig.add_subplot(gs[:3,2], sharey=ax1)
    ax3 = fig.add_subplot(gs[3:,:])
    fac=0
    colors = pl.cm.RdPu(np.linspace(0,1,len(inds)))
    for ii, ind in enumerate(inds):
        d0=np.abs(x_data[ind,110:156])
        d1=np.abs(x_data[ind,366:412])
        d2=np.abs(x_data[ind,622:668])
        n0=n_data[ind,110:156]
        n1=n_data[ind,366:412]
        n2=n_data[ind,622:668]
        ax0.plot(t,d0+n0+fac*ii,color=colors[ii])  
        ax1.plot(t,d1+n1+fac*ii,color=colors[ii])   
        ax2.plot(t,d2+n2+fac*ii,color=colors[ii],label=str(ii+1)) 
        # ax0.plot(t,d0+n0+fac*ii,color=(0.6,0.6,0.6))  
        # ax1.plot(t,d1+n1+fac*ii,color=(0.6,0.6,0.6))    
        # ax2.plot(t,d2+n2+fac*ii,color=(0.6,0.6,0.6)) 
    ax0.set_xlim((0,45))
    ax0.set_facecolor((0.75,0.75,0.75))
    ax0.text(2,1.42,'A',fontsize=24,weight='bold')
    ax0.set_title('North')
    ax1.set_xlim((0,45))
    ax1.set_facecolor((0.75,0.75,0.75))
    ax1.text(2,1.42,'B',fontsize=24,weight='bold')
    ax1.set_title('East')
    ax2.set_xlim((0,45))
    ax2.set_facecolor((0.75,0.75,0.75))
    ax2.text(2,1.42,'C',fontsize=24,weight='bold')
    ax2.set_title('Vertical')
    ax2.legend(loc="best",ncol=2)
    ax3.hist(pgd,bins=np.arange(0,5,0.1), color=(162/256,210/256,255/256),alpha=0.5, edgecolor='black', linewidth=1.2)
    ax3.set_xlim((0,5))
    ax3.set_ylim((0,65000))
    ax3.text(4.75,57000,'D',fontsize=24,weight='bold')
    #ax0.plot([5, 5],[0.7, 0.9],color='black')
    #ax0.text(7,0.8,'20 cm',rotation=0,fontsize=14 )
    ax0.set_ylabel('Amplitude (m)',fontsize=14)
    ax1.set_xlabel('Time (s)',fontsize=14)
    ax3.set_xlabel('PGD (m)',fontsize=14)
    ax3.set_ylabel('Frequency',fontsize=14)
    
    fig.savefig("figure2.png", dpi=300)