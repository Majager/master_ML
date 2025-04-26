from hmmlearn.hmm import GMMHMM
import numpy as np
import pickle
import datetime
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
import machine_learning
from statistics import mode

import os
os.environ['OMP_NUM_THREADS'] = '1'

class HMM(ClassifierMixin, BaseEstimator):
    def __init__(self, model_arcitechture):
        self.model_arcitechture = model_arcitechture
        self.models = {}
        self.feature_importances_ = []

    # Find weights of the features, to find which are most important
    def compute_feature_importance(self, X, y,lengths):
        base_score = self.score(X,y)
        n_features = X.shape[1]
        importances = []
        for i in range(n_features):
            X_reduced = np.delete(X, i, axis=1)
            self.fit(X_reduced,y,lengths,False)
            reduced_score = self.score(X_reduced,y)
            importances.append(base_score - reduced_score)
        self.feature_importances_ = np.array(importances)
        print(self.feature_importances_)
    
    # Fit HMM model to given training data and parameters
    # X on format (n_samples,) where each sample is of length (n_features)
    def fit(self, X, y, lengths = None, calculate_importance=False):
        self.classes = np.unique(y)
        for class_label in self.classes:
            [n_c, n_m] = self.model_arcitechture[class_label]
            model = GMMHMM(n_components=n_c,n_mix=n_m,algorithm="viterbi")
            class_data = [X[idx] for idx in range(len(y)) if y[idx] == class_label]
            length = None if lengths is None else lengths[class_label]
            model.fit(class_data, length)
            self.models[class_label] = model
        if calculate_importance:
            self.compute_feature_importance(X, y, lengths)
        return self
    
    # Predicts value for segment
    def _predict_segments(self, segments_data, n_features):
        segments_predictions = []
        for recording in segments_data:
            recording_predictions = []
            for segment in recording:
                segment = segment.reshape(-1, n_features)
                scores = {label: model.score(segment) for label, model in self.models.items()}
                predicted_label = max(scores, key=scores.get)
                recording_predictions.append(predicted_label)
            segments_predictions.append(recording_predictions)
        return segments_predictions
    
    # Estimate probability of belonging to the meal model
    def _predict_segments_proba(self, segments_data, n_features):
        segments_predictions_proba = []
        for recording in segments_data:
            subject_predictions_proba = []
            for segment in recording:
                segment = segment.reshape(-1, n_features)
                meal_score = self.models[1].score(segment)
                subject_predictions_proba.append(np.exp(meal_score))
            segments_predictions_proba.append(subject_predictions_proba)
        return segments_predictions_proba
    
    # Divides into subsequences and predicts label per subsequence
    def predict(self, X, lengths = None, segment_parameter = 50):
        original_format_data, _ = split_test_data(X,None,lengths)
        segments_data, _ = segment_data(original_format_data,None,segment_parameter)
        return self._predict_segments(segments_data, X.shape[1])
    
    def predict_proba(self, X, lengths = None, segment_parameter = 50):
        original_format_data, _ = split_test_data(X,None,lengths)
        segments_data, _ = segment_data(original_format_data,None,segment_parameter)
        return self._predict_segments_proba(segments_data, X.shape[1])
    
    # Returns the mean accuracy on the given test data and labels
    def score(self, X, y, lengths = None, segment_parameter = 50):
        original_format_data, original_format_label = split_test_data(X,y,lengths)
        segments_data, segments_labels = segment_data(original_format_data,original_format_label,segment_parameter)
        segments_predictions = self._predict_segments(segments_data,X.shape[1])
        segments_labels_concatenated = np.concatenate(segments_labels, axis=0)
        segments_predictions_concatenated = np.concatenate(segments_predictions,axis=0)
        f1 = f1_score(y_true=segments_labels_concatenated, y_pred=segments_predictions_concatenated, average='macro')
        return f1

# Segments data into different segments for testing
# Data is in format (n_recordings, n_segments, n_features)
def segment_data(data, labels = None , n_segments = 50, step_size = 1):
    # Find number of observations in the data
    segmented_data = []
    segmented_labels = []

    for i, recording in enumerate(data):
        recording_segmented_data, recording_segmented_labels = [], []
        for segment_start_idx in range(0,len(recording)-n_segments,step_size):
            segment = recording[(segment_start_idx):(segment_start_idx+n_segments)]  
            # Find index in labels based on which segment from specific recording
            recording_segmented_data.append(segment)
            # Majority voting for the separate segments
            if labels != None:
                # Extract most common label for the entire segment and use as label
                labels_segment = int(mode(labels[i][segment_start_idx:(segment_start_idx+n_segments)]))
                recording_segmented_labels.append(labels_segment)
        segmented_data.append(recording_segmented_data)
        segmented_labels.append(recording_segmented_labels)

    return segmented_data, segmented_labels

# Prepare data for format to be used in HMM model
# X should be the feature matrix of individual samples, with shape (n_samples, n_features)
# lengths represents length of the individual sequences and sums to n_samples, with shape (n_sequences,)
def prepare_train_data(data, labels):
    meal_lengths = []
    nonmeal_lengths = []

    for idx, recording_data in enumerate(data):
        meal_found, idx_meal_start, idx_meal_end = False, 0, -1
        for idx_segment, segment in enumerate(labels[idx]):
            if (segment == 1 and meal_found == False):
                idx_meal_start = idx_segment
                meal_found = True
            elif (segment == 0 and meal_found == True):
                idx_meal_end = idx_segment
                break
        if meal_found == True:
            meal_lengths.append(idx_meal_end-idx_meal_start)
            nonmeal_lengths.append(idx_meal_start)
            nonmeal_lengths.append(len(recording_data)-idx_meal_end)
        else:
            nonmeal_lengths.append(len(recording_data))

    concatenated_data = np.vstack(np.concatenate(data,axis=0))
    concatenated_labels = np.vstack(np.concatenate(labels,axis=0))
    return concatenated_data, concatenated_labels, [nonmeal_lengths,meal_lengths]

# Prepare data for format to be used in HMM model
# X should be the feature matrix of individual samples, with shape (n_samples, n_features)
# lengths represents length of the individual sequences and sums to n_samples, with shape (n_sequences,)
def prepare_test_data(data, labels):
    lengths = []
    for recording in data:
        lengths.append(len(recording))
    concatenated_data = np.vstack(np.concatenate(data,axis=0))
    concatenated_labels = np.concatenate(labels,axis=0)
    return concatenated_data, concatenated_labels, lengths

def split_test_data(concatenated_data, concatenated_labels = None, lengths = None):
    data_split = []
    labels_split = []
    start = 0

    if lengths is None:
        data_split.append(concatenated_data)
        if concatenated_labels.any():
            labels_split.append(concatenated_labels)
    else:
        for length in lengths:
            end = start + length
            data_split.append(concatenated_data[start:end])
            if concatenated_labels is not None:
                labels_split.append(concatenated_labels[start:end])
            start = end
    return data_split, labels_split

# Train hidden markov models based on data
def train_manager(hmm, data, labels):
    #  Prepare data for use in the HMM
    data, labels, lengths = prepare_train_data(data, labels)
    hmm = hmm.fit(data,labels,lengths)
    return hmm

def test_manager(hmm, data, labels, n_segments = 50):
    segments_data, segments_labels = segment_data(data,labels,n_segments)
    data, labels, lengths = prepare_test_data(data, labels)
    segments_predictions = hmm.predict(data, lengths,n_segments)
    segments_predictions_proba = hmm.predict_proba(data, lengths, n_segments)
    return segments_labels, segments_predictions, segments_predictions_proba

# Function to create HMM models, train and test
# segment_parameters in input is given by [segment_length,overlap_length,n_segments]
def run_HMM_model_train_and_validation(data, labels, recording_ids, test_name, model_arcitechture, segment_parameters): #k_folds
    # Timestamp of cross-validation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Cross-validation loop to be able to average over all folds
    kfold = KFold(n_splits=10, shuffle=True)
    for fold, (train_idx, validation_idx) in enumerate(kfold.split(data)):
        # Split data
        train_data, train_labels, _, validation_data, validation_labels, validation_recording_ids = machine_learning.split_data(data,labels,recording_ids,validation_idx)
        
        # Train HMM  based on train set of this fold
        hmm = HMM(model_arcitechture)
        hmm = train_manager(hmm,train_data,train_labels)

        # Test HMM model based on the validation fold of this iteration
        true, predictions, predictions_proba = test_manager(hmm=hmm,data=validation_data,labels=validation_labels,n_segments=segment_parameters[2])
        
        # Store results to pickle file
        r_path = machine_learning.store_results_filename(test_name,timestamp)
        full_path = os.path.join(r_path,f"fold{fold+1}.pickle")
        with open(full_path,'wb') as handle:
            pickle.dump([true,predictions,predictions_proba,validation_recording_ids,segment_parameters],handle,protocol=pickle.HIGHEST_PROTOCOL)
    machine_learning.store_parameters(test_name, ["test"])

def run_HMM_model_test(train_data, train_labels, train_recording_ids, test_data, test_labels, test_recording_ids,test_name,model_arcitechture,segment_parameters):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Train HMM  based on train set of this fold
    hmm = HMM(model_arcitechture)
    hmm = train_manager(hmm,train_data,train_labels)

    true, predictions, predictions_proba = test_manager(hmm=hmm,data=test_data,labels=test_labels,n_segments=segment_parameters[2])
        
    # Store results to pickle file
    r_path = machine_learning.store_results_filename(test_name,timestamp)
    full_path = os.path.join(r_path,f"test.pickle")
    with open(full_path,'wb') as handle:
        pickle.dump([true,predictions,predictions_proba,test_recording_ids,segment_parameters],handle,protocol=pickle.HIGHEST_PROTOCOL)
    machine_learning.store_parameters(test_name, ["test"])