from hmmlearn.hmm import GMMHMM
import numpy as np
import pickle
import datetime
import random
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import machine_learning

import os
os.environ['OMP_NUM_THREADS'] = '1'

# Split data into separate groups of data from meal- and nonmealdata
def split_to_mealdata_and_nonmealdata(data,labels):
    num_subjects = len(data)
    meal_data = []
    nonmeal_data = []
    # Find index of where the meal is for each subject and add to the corresponding dataset
    for idx, subject_data in enumerate(data):
        meal_found = False
        idx_meal_start = 0
        idx_meal_end = -1
        for idx_meal, segment in enumerate(labels[idx]):
            if (segment == 1 and meal_found == False):
                idx_meal_start = idx_meal
                meal_found = True
            elif (segment == 0 and meal_found == True):
                idx_meal_end = idx_meal
                break
        if meal_found == True:
            meal_data.append(subject_data[idx_meal_start:idx_meal_end])
            nonmeal_data.append(subject_data[:idx_meal_start]) 
            nonmeal_data.append(subject_data[idx_meal_end:])
        else:
            nonmeal_data.append(subject_data)
    return meal_data, nonmeal_data

# Prepare data for format to be used in HMM model
# X should be the feature matrix of individual samples, with shape (n_samples, n_features)
# lengths represents length of the individual sequences and sums to n_samples, with shape (n_sequences,)
def prepare_data(data):
    lengths = []
    for subject in data:
        # Find the length of each the different segments for each subject
        lengths.append(len(subject))
    concatenated_data = np.vstack(np.concatenate(data,axis=0))
    return concatenated_data, lengths

# Segments data into different segments for testing
def segment_data(data, lengths, labels, segment_size):
    # Find number of observations in the data
    segmented_data = []
    segmented_labels = []

    start_idx = 0
    for i, subject_length in enumerate(lengths):
        subject_segments_length = subject_length//segment_size
        subject_segmented_data = []
        subject_segmented_labels = []
        for j in range(0,subject_length-segment_size):
            # Find index based on which segment from specific subject
            # Find a section of the data and divide in to separate components of the vector
            segment_start_idx = j
            segment_end_idx = j+segment_size
            segment = data[(start_idx+segment_start_idx):(start_idx+segment_end_idx)]  
            # Find index in labels based on which segment from specific subject
            subject_segmented_data.append(segment)
            # Majority voting for the separate segments
            labels_segment = round(np.mean(labels[i][segment_start_idx:segment_end_idx]))
            subject_segmented_labels.append(labels_segment)
        segmented_data.append(subject_segmented_data)
        segmented_labels.append(subject_segmented_labels)
        start_idx += subject_length
    
    return segmented_data, segmented_labels

# Train hidden markov models based on data
def train_manager(model_meal,model_nonmeal,data,labels):
    #  Prepare data for use in the HMM
    data_meal, data_nonmeal = split_to_mealdata_and_nonmealdata(data,labels)
    meal_data, meal_lengths = prepare_data(data_meal)
    nonmeal_data, nonmeal_lengths = prepare_data(data_nonmeal)

    # Train meal model based on meal data from training set
    model_meal = model_meal.fit(meal_data,meal_lengths)

    # Train nonmeal model based on nonmeal data from training set
    model_nonmeal = model_nonmeal.fit(nonmeal_data,nonmeal_lengths)

    return model_meal, model_nonmeal

# Test the hidden markov models based on data
def test_manager(model_meal, model_nonmeal,data,labels,n_segments):
    validation_data, lengths = prepare_data(data)
    segments_data, segments_labels = segment_data(validation_data,lengths,labels,n_segments)
    
    segments_predictions = []
    segments_predictions_proba = []
    for subject_idx in range(len(segments_data)):
        subject_predictions = []
        subject_predictions_proba = []
        for segment in segments_data[subject_idx]:
            segment = segment.reshape(-1, validation_data.shape[1])
            meal_score = model_meal.score(segment)
            nonmeal_score = model_nonmeal.score(segment)  
            if meal_score > nonmeal_score:
                subject_predictions.append(1)
            else:
                subject_predictions.append(0)
            subject_predictions_proba.append(np.exp(meal_score))
        segments_predictions.append(subject_predictions)
        segments_predictions_proba.append(subject_predictions_proba)

    return segments_labels, segments_predictions, segments_predictions_proba

# Function to create HMM models, train and test
# segment_parameters in input is given by [segment_length,overlap_length,n_segments]
def run_HMM_model_train_and_validation(data, labels, recording_ids, test_name, model_arcitechture, segment_parameters): #k_folds
    # Timestamp of cross-validation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    [n_mix_meal, n_components_meal, n_mix_nonmeal, n_components_nonmeal] = model_arcitechture

    # Cross-validation loop to be able to average over all folds
    kfold = KFold(n_splits=len(data), shuffle=True)
    for fold, (train_idx, validation_idx) in enumerate(kfold.split(data)):
        # Split data
        train_data, train_labels, _, validation_data, validation_labels, validation_recording_ids = machine_learning.split_data(data,labels,recording_ids,validation_idx)
        
        # Train HMM models based on train set of this fold
        model_meal = GMMHMM(n_components=n_components_meal,n_mix=n_mix_meal,algorithm="viterbi")
        model_nonmeal = GMMHMM(n_components=n_components_nonmeal, n_mix=n_mix_nonmeal, algorithm="viterbi")
        model_meal, model_nonmeal = train_manager(model_meal=model_meal,model_nonmeal=model_nonmeal,data=train_data,labels=train_labels)
        
        # Test HMM model based on the validation fold of this iteration
        true, predictions, predictions_proba = test_manager(model_meal=model_meal,model_nonmeal=model_nonmeal,data=validation_data,labels=validation_labels,n_segments=segment_parameters[2])
        
        # Store results to pickle file
        r_path = machine_learning.store_results_filename(test_name,timestamp)
        full_path = os.path.join(r_path,f"fold{fold+1}.pickle")
        with open(full_path,'wb') as handle:
            pickle.dump([true,predictions,predictions_proba,validation_recording_ids,segment_parameters],handle,protocol=pickle.HIGHEST_PROTOCOL)

def run_HMM_model_test(data, labels, recording_ids,test_name,model_arcitechture,segment_parameters):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    [n_mix_meal, n_components_meal, n_mix_nonmeal, n_components_nonmeal] = model_arcitechture
    train_data, train_labels, train_recording_ids, test_data, test_labels, test_recording_ids = machine_learning.split_data(data,labels,recording_ids,[0])
    
    model_meal = GMMHMM(n_components=n_components_meal,n_mix=n_mix_meal,algorithm="viterbi")
    model_nonmeal = GMMHMM(n_components=n_components_nonmeal, n_mix=n_mix_nonmeal, algorithm="viterbi")
    model_meal, model_nonmeal = train_manager(model_meal=model_meal,model_nonmeal=model_nonmeal,data=train_data,labels=train_labels)
    
    true, predictions, predictions_proba = test_manager(model_meal=model_meal,model_nonmeal=model_nonmeal,data=test_data,labels=test_labels,n_segments=segment_parameters[2])
        
    # Store results to pickle file
    r_path = machine_learning.store_results_filename(test_name,timestamp)
    full_path = os.path.join(r_path,f"test.pickle")
    with open(full_path,'wb') as handle:
        pickle.dump([true,predictions,predictions_proba,test_recording_ids,segment_parameters],handle,protocol=pickle.HIGHEST_PROTOCOL)
