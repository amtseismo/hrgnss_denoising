# GNSS Denoising

## Description

This repository contains data, training scripts, model architectures and other utilities for creating a U-shaped Convolutional Neural Network trained to denoise high-rate global navigation satellite system data (HR-GNSS).  Descriptions of all files and scripts can be found below.  Further details are available in:

Thomas, A. M., D. Melgar, S. Dybing, and J. Searcy (202?) Deep learning for denoising High-Rate Global Navigation Satellite System data. Submitted to Seismica.

BibTeX:

    @article{thomas202Xhrgnss,
        title={Deep learning for denoising High-Rate Global Navigation Satellite System data},
        author={Thomas, Amanda M and Melgar, Diego, and Dybing, Sydney and Searcy, Jacob},
        journal={Seismica},
    }

Zenodo link here

## Requirements

In order to run the scripts you will need [Obspy](https://docs.obspy.org/) (I have version 1.3.0), [Tensorflow](https://www.tensorflow.org/) (I have version 2.8.1), Keras (2.8.0), and other standard python utilities such as numpy, scipy, and matplotlib.  I recommend creating a [conda](https://docs.conda.io/en/latest/) environment and installing packages into it.    

## File Descriptions
* gnss_denoiser_3comp_v1.py - training script for model 1 (described in above reference) 
* gnss_denoiser_3comp_v2.py - training script for model 2 (described in above reference) 
* gnss_denoiser_3comp_v3.py - training script for model 3 (described in above reference) 
* gnss_tools.py - extra utilities for unets, scripts to calculate performance metrics, etc.
* nd3_data_subset.hdf5 - fakequakes synthetic earthquakes, N=10000
* 729k_noise_subset.hdf5 - noise samples, N=10000
* softlayers_subset.hdf5 - fakequakes synthetic earthquakes with soft layers, N=10000

## File usage
The gnss_denoiser_3comp_vX.py scripts are the drivers.  Each script has a # SET OPTIONS section in the header with the following options:  
* Setting Train=True in the file header will train a new model with output name quickie_3comp_norm_input_v[X]_[fac]_[month]-[day].tf where X is the model version, fac is the model size set in the header of the gnss_denoiser_3comp_vX.py script, month and day correspond to the date the model began training
* plots makes various plots that were useful in developing the models (True)
* epos is the number of epochs to train for (50)
* fac is the model size (2 or 4 are used in the manuscript)
* sr is the sample rate (1)
* eps is epsilon (1e-9)
* drop is the model drop rate (0.2)
* nlen is window length in samples (128)
* nperseg is a parameter for STFT (31)
* noverlap is a parameter for STFT (30)
* norm_input normalizes input -- leave this alone (True)