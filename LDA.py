import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import datetime
import os
import pickle
import time
from sklearn.model_selection import KFold
import machine_learning

def revert_to_meal_or_nomeal(true, predictions):
    for recording_idx in range(len(true)):
        for segment_idx, segment_value in enumerate(true[recording_idx]):
            if segment_value > 1:
                true[recording_idx][segment_idx] = 0
    
    for recording_idx in range(len(predictions)):
        for segment_idx, segment_value in enumerate(predictions[recording_idx]):
            if segment_value > 1:
                predictions[recording_idx][segment_idx] = 0

    return true,predictions

# Function to create the LDA model
def LDA_probability_analysis(data, labels, recording_ids, test_name, segment_parameters, multiclass): #k_folds
    segment_parameters[2] = 1

    thresholds = []
    timestamps = []
    for i in range(1,100):
        thresholds.append(i/100)
        timestamps.append(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        time.sleep(2)

    # Cross-validation loop to be able to average over all folds
    kfold = KFold(n_splits=10, shuffle=True)
    for fold, (train_idx, validation_idx) in enumerate(kfold.split(data)):
        # Split data
        train_data, train_labels, _, validation_data, validation_labels, validation_recording_ids = machine_learning.split_data(data,labels,recording_ids,validation_idx)
        
        # Train LDA
        classifier = LinearDiscriminantAnalysis(solver="eigen",shrinkage=0.9)
        train_data, train_labels = np.concatenate(train_data,axis=0), np.concatenate(train_labels,axis=0)
        classifier.fit(train_data, train_labels)

        meal_probabilities = []
        predictions_proba = []
        for idx in range(len(validation_data)):
            predictions_proba_recording = classifier.predict_proba(validation_data[idx])
            meal_probabilities_recording = predictions_proba_recording[:,1]
            predictions_proba.append(predictions_proba_recording)
            meal_probabilities.append(meal_probabilities_recording)

        for idx, threshold in enumerate(thresholds):
            predictions = [[int(p > threshold) for p in recording] for recording in meal_probabilities]

            if multiclass:
                validation_labels,predictions = revert_to_meal_or_nomeal(validation_labels,predictions)

            # Store results to pickle file
            r_path = machine_learning.store_results_filename(test_name,timestamps[idx])
            full_path = os.path.join(r_path,f"fold{fold+1}.pickle")
            with open(full_path,'wb') as handle:
                pickle.dump([validation_labels,predictions,predictions_proba,validation_recording_ids,segment_parameters],handle,protocol=pickle.HIGHEST_PROTOCOL)            
    
    machine_learning.store_parameters(test_name, thresholds)
        
# Function to create the LDA model
def run_LDA_train_and_validation(data, labels, recording_ids, test_name, segment_parameters, multiclass): #k_folds
    segment_parameters[2] = 1
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Cross-validation loop to be able to average over all folds
    kfold = KFold(n_splits=10, shuffle=True)
    for fold, (train_idx, validation_idx) in enumerate(kfold.split(data)):
        # Split data
        train_data, train_labels, _, validation_data, validation_labels, validation_recording_ids = machine_learning.split_data(data,labels,recording_ids,validation_idx)
        
        # Train LDA
        classifier = LinearDiscriminantAnalysis(solver="eigen",shrinkage=0.9)
        train_data, train_labels = np.concatenate(train_data,axis=0), np.concatenate(train_labels,axis=0)
        classifier.fit(train_data, train_labels)

        predictions = []
        predictions_proba = []
        for idx in range(len(validation_data)):
            predictions.append(classifier.predict(validation_data[idx]))
            predictions_proba.append(classifier.predict_proba(validation_data[idx]))

        if multiclass:
            validation_labels,predictions = revert_to_meal_or_nomeal(validation_labels,predictions)

        # Store results to pickle file
        r_path = machine_learning.store_results_filename(test_name,timestamp)
        full_path = os.path.join(r_path,f"fold{fold+1}.pickle")
        with open(full_path,'wb') as handle:
            pickle.dump([validation_labels,predictions,predictions_proba,validation_recording_ids,segment_parameters],handle,protocol=pickle.HIGHEST_PROTOCOL)            
        time.sleep(2)
    
    machine_learning.store_parameters(test_name, ["test"])

def train_test(train_data, train_labels, train_recording_ids, test_data, test_labels, test_recording_ids,test_name,segment_parameters):
    segment_parameters[2] = 1
    classifier = LinearDiscriminantAnalysis(solver="eigen",shrinkage=0.9)
    train_data, train_labels = np.concatenate(train_data,axis=0), np.concatenate(train_labels,axis=0)
    classifier.fit(train_data, train_labels)

    predictions = []
    predictions_proba = []
    for idx in range(len(test_data)):
        predictions.append(classifier.predict(test_data[idx]))
        predictions_proba.append(classifier.predict_proba(test_data[idx]))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Store results to pickle file
    r_path = machine_learning.store_results_filename(test_name,timestamp)
    full_path = os.path.join(r_path,f"test.pickle")
    with open(full_path,'wb') as handle:
        pickle.dump([test_labels,predictions,predictions_proba,test_recording_ids,segment_parameters],handle,protocol=pickle.HIGHEST_PROTOCOL)
    
    machine_learning.store_parameters(test_name, ["test"])