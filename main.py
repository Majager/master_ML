import signal_processing
import file_convert
import numpy as np
import run
import feature_selection

def main():
    # Parameters
    sr = 22050                  # Sampling rate of signal reading, None corresponds to true value
    segment_length = 1          # Time in seconds for each segment
    overlap_length = 0.5*segment_length        # Time in seconds for overlap
    n_fft = 2048                # Number of samples in NFFT for calculating features    
    hop_length = 512            # Number of samples in hop length for features
    n_mfcc = 39                 # Number of features to extract from MFCC
    n_mix_meal = 7              # Number of mixtures in the GMMHMM for meal model
    n_components_meal = 5       # Number of hidden states in the model for meal model
    n_mix_nonmeal = 5           # Number of mixtures in the GMMHMM for nonmeal model
    n_components_nonmeal = 9    # Number of hidden states in the model for nonmeal model
    n_segments = 70             # Number of segments in a sub sequence for testing
    update_features = False     # Bool of whether or not to update features, or use previous result from function
    model = "LDA"    # Machine learning method to use, LDA/HMM/RNN/L_P
    validation_set = True       # True tests on validation set, while False uses test set
    multiclass = False           # Several classes/ meal and non-meal
    loso = False                # Parameter to decide if fold will be based on randomized between recordings or leave one subject out

    # Generate test name based on parameters
    suffix = "validation" if validation_set else "test" 
    multiclass_suffix ="_multiclass" if multiclass else ""
    test_name = f"{model}_{suffix}{multiclass_suffix}"

    model_architecture = {0:[n_components_nonmeal, n_mix_nonmeal], 1:[n_components_meal, n_mix_meal]}
    
    # Find recorded data
    root_folder_path = 'C:\\Users\\MajaE\\src\\repos\\master_ML\\Data'
    position = 'pos1'
    wav_files, metadata_files, file_ids = file_convert.find_data(root_folder_path,position)
    
    # Feature extraction from recorded data
    features, labels, recording_ids = signal_processing.feature_extraction(wav_files, metadata_files, file_ids, position, segment_length, overlap_length, sr,n_fft,hop_length,n_mfcc,update_features,multiclass)
    
    # Feature selection
    features = feature_selection.feature_selection_extraction(features,model)

    # Run test
    run.run_test(model,validation_set,test_name,features,labels,recording_ids,model_architecture,[segment_length,overlap_length,n_segments], multiclass,loso)
    
main()