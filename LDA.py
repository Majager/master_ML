import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import datetime
import os
import pickle
import time
from sklearn.model_selection import KFold

def split_data_to_train_and_test(data,labels,train_indices,test_indices,recording_ids):
    test_data = [data[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    test_recording_ids = [recording_ids[i] for i in test_indices]

    # The rest of the data will be in the training set
    train_data = [data[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_recording_ids = [recording_ids[i] for i in train_indices]
    return train_data, train_labels, train_recording_ids, test_data, test_labels, test_recording_ids

# Split the data set into training and validation set based on indices
def split_data_to_train_and_validation(data,labels,recording_ids,train_indices,validation_indices):
    validation_data = [data[i] for i in validation_indices]
    validation_labels = [labels[i] for i in validation_indices]
    validation_recording_ids = [recording_ids[i] for i in validation_indices]

    # The rest of the data will be in the training set
    train_data = [data[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]

    return train_data, train_labels, validation_data, validation_labels, validation_recording_ids


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
        train_data, train_labels, validation_data, validation_labels, validation_recording_ids = split_data_to_train_and_validation(data,labels,recording_ids,train_idx,validation_idx)
        
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
            validation_path = f"Results\\{test_name}"
            if not os.path.exists(validation_path):
                os.mkdir(validation_path)
            data_path = os.path.join(validation_path,"data")
            if not os.path.exists(data_path):
                os.mkdir(data_path)
            cv_path = os.path.join(data_path,timestamp[idx])
            if not os.path.exists(cv_path):
                os.mkdir(cv_path)
            full_path = os.path.join(cv_path,f"fold{fold+1}.pickle")
            with open(full_path,'wb') as handle:
                pickle.dump([true_segmented,predictions_segmented,validation_recording_ids,segment_parameters],handle,protocol=pickle.HIGHEST_PROTOCOL)
            
            print("Parameter done", segment_size)
            time.sleep(2)
    

def train_test(features,labels,recording_ids,test_name,segment_parameters):
    classifier = LinearDiscriminantAnalysis()

    train_data, train_labels, train_recording_ids, test_data, test_labels, test_recording_ids = split_data_to_train_and_test(features,labels,[1,2,3],[0],recording_ids)
    
    train_data, train_labels = np.concatenate(train_data,axis=0), np.concatenate(train_labels,axis=0)
    classifier.fit(train_data, train_labels)

    predictions = []
    for idx in range(len(test_data)):
        predictions.append(classifier.predict(test_data[idx]))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    true_segmented, predictions_segmented = segment_labels(test_labels,predictions,segment_parameters[2])
            
    # Store results to pickle file
    test_path = f"Results\\{test_name}"
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    data_path = os.path.join(test_path,"data")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    t_path = os.path.join(data_path,timestamp)
    if not os.path.exists(t_path):
        os.mkdir(t_path)
    full_path = os.path.join(t_path,f"test.pickle")
    with open(full_path,'wb') as handle:
        pickle.dump([true_segmented,predictions_segmented,test_recording_ids,segment_parameters],handle,protocol=pickle.HIGHEST_PROTOCOL)

