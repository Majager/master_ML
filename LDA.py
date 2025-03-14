import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import datetime
import os
import pickle
import time
from sklearn.model_selection import KFold
import machine_learning

# Segments data into different segments for testing
def segment_labels(true, predictions, segment_size):
    # Find number of observations in the data
    segmented_true = []
    segmented_predictions = []

    for i in range(len(true)):
        number_subsequences = len(true[i])//segment_size
        subject_segmented_true = []
        subject_segmented_predictions = []
        for j in range(0,number_subsequences):
            # Majority voting for the separate segments
            subsequence_true = round(np.mean(true[i][(j*segment_size):((j+1)*segment_size)]))
            subsequence_predictions = round(np.mean(predictions[i][(j*segment_size):((j+1)*segment_size)]))
            subject_segmented_true.append(subsequence_true)
            subject_segmented_predictions.append(subsequence_predictions)
        segmented_true.append(subject_segmented_true)
        segmented_predictions.append(subject_segmented_predictions)
    return segmented_true,segmented_predictions

# Function to create the LDA model
def run_LDA_train_and_validation(data, labels, recording_ids, test_name, segment_parameters): #k_folds
    # Timestamp of cross-validation
    segment_sizes = [1,10,20,30,40,50,60,70,80,90,100]
    timestamp = []
    for i in range (len(segment_sizes)):
        timestamp.append(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        time.sleep(1)

    # Cross-validation loop to be able to average over all folds
    kfold = KFold(n_splits=len(data), shuffle=True)
    for fold, (train_idx, validation_idx) in enumerate(kfold.split(data)):
        # Split data
        train_data, train_labels, _, validation_data, validation_labels, validation_recording_ids = machine_learning.split_data(data,labels,recording_ids,validation_idx)
        
        # Train LDA
        classifier = LinearDiscriminantAnalysis()
        train_data, train_labels = np.concatenate(train_data,axis=0), np.concatenate(train_labels,axis=0)
        classifier.fit(train_data, train_labels)

        predictions = []
        for idx in range(len(validation_data)):
            predictions.append(classifier.predict(validation_data[idx]))

        for idx, segment_size in enumerate(segment_sizes):
            true_segmented, predictions_segmented = segment_labels(validation_labels,predictions,segment_size)

            segment_parameters[2]=segment_size
        
            # Store results to pickle file
            r_path = machine_learning.store_results_filename(test_name,timestamp[idx])
            full_path = os.path.join(r_path,f"fold{fold+1}.pickle")
            with open(full_path,'wb') as handle:
                pickle.dump([true_segmented,predictions_segmented,validation_recording_ids,segment_parameters],handle,protocol=pickle.HIGHEST_PROTOCOL)
            print("Parameter done", segment_size)
            time.sleep(2)
    

def train_test(features,labels,recording_ids,test_name,segment_parameters):
    classifier = LinearDiscriminantAnalysis()

    train_data, train_labels, train_recording_ids, test_data, test_labels, test_recording_ids = machine_learning.split_data(features,labels,recording_ids,[1,2,3],[0])
    
    train_data, train_labels = np.concatenate(train_data,axis=0), np.concatenate(train_labels,axis=0)
    classifier.fit(train_data, train_labels)

    predictions = []
    for idx in range(len(test_data)):
        predictions.append(classifier.predict(test_data[idx]))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    true_segmented, predictions_segmented = segment_labels(test_labels,predictions,segment_parameters[2])
            
    # Store results to pickle file
    r_path = machine_learning.store_results_filename(test_name,timestamp)
    full_path = os.path.join(r_path,f"test.pickle")
    with open(full_path,'wb') as handle:
        pickle.dump([true_segmented,predictions_segmented,test_recording_ids,segment_parameters],handle,protocol=pickle.HIGHEST_PROTOCOL)
