import signal_processing
import hmm
import rnn
import LDA
import file_convert
import random
import matplotlib.pyplot as plt
from hmmlearn.hmm import GMMHMM
import glob
import os
import numpy as np
import machine_learning
import l_predict
import feature_selection

def main():
    # Constants
    sr = 22050                  # Sampling rate of signal reading, None corresponds to true value
    segment_length = 1          # Time in seconds for each segment
    overlap_length = 0.5        # Time in seconds for overlap
    n_fft = 2048                # Number of samples in NFFT for calculating features    
    hop_length = 512            # Number of samples in hop length for features
    n_mfcc = 20                 # Number of features to extract from MFCC
    update_features = False     # Bool of whether or not to update features, or use previous result from function
    n_mix_meal = 3              # Number of mixtures in the GMMHMM for meal model
    n_components_meal = 5       # Number of hidden states in the model for meal model
    n_mix_nonmeal = 6           # Number of mixtures in the GMMHMM for nonmeal model
    n_components_nonmeal = 7    # Number of hidden states in the model for nonmeal model
    test_name = "l_predict"   # Test name 
    n_segments = 70            # Number of segments in a sub sequence for testing
    
    # Find recorded data
    root_folder_path = 'C:\\Users\\MajaE\\src\\repos\\master_ML\\Recordings'
    position = 'pos1-pos1m'
    wav_files, metadata_files, recording_ids = file_convert.find_data(root_folder_path,position)
    
    # Feature extraction from recorded data
    features, labels = signal_processing.feature_extraction(wav_files, metadata_files, position, segment_length, overlap_length, sr,n_fft,hop_length,n_mfcc,update_features)
    
    # Running machine learning on features
    train_data, train_labels, train_recording_ids, test_data, test_labels, test_recording_ids = machine_learning.split_data(features,labels,recording_ids,[0])
    
    # Feature selection
    feature_selection.feature_selection_LDA(train_data, train_labels)

    # HMM code
    #hmm.run_HMM_model_train_and_validation(data=train_data, labels=train_labels, recording_ids=train_recording_ids,test_name=test_name,model_arcitechture=[n_mix_meal, n_components_meal, n_mix_nonmeal, n_components_nonmeal],segment_parameters=[segment_length,overlap_length,n_segments])
    #hmm.run_HMM_model_test(data=features,labels=labels,recording_ids=recording_ids,test_name=test_name,model_arcitechture=[n_mix_meal, n_components_meal, n_mix_nonmeal, n_components_nonmeal],segment_parameters=[segment_length,overlap_length,n_segments])

    # RNN code
    #rnn.train_test(features,labels,recording_ids,test_name,[segment_length,overlap_length,n_segments])

    #LDA
    #LDA.run_LDA_train_and_validation(train_data,train_labels,train_recording_ids,test_name,[segment_length,overlap_length,n_segments])
    #LDA.train_test(features,labels,recording_ids,test_name,[segment_length,overlap_length,n_segments])

    # Lazypredict
    #l_predict.train_test(train_data,train_labels,test_data,test_labels)

main()