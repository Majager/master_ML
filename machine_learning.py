import os
import numpy as np
import json
from statistics import mode

# Split the data set into first and second set based on indices
def split_data(data,labels,recording_ids,second_indices):
    first_indices = [j for j in range(len(data)) if j not in set(second_indices)]
    first_data = [data[i] for i in first_indices]
    first_labels = [labels[i] for i in first_indices]
    first_recording_ids = [recording_ids[i] for i in first_indices]

    second_data = [data[i] for i in second_indices]
    second_labels = [labels[i] for i in second_indices]
    second_recording_ids = [recording_ids[i] for i in second_indices]
    return first_data, first_labels, first_recording_ids, second_data, second_labels, second_recording_ids

def store_results_filename(test_name,timestamp):
    results_path = f"Results\\{test_name}"
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    data_path = os.path.join(results_path,"data")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    r_path = os.path.join(data_path,timestamp)
    if not os.path.exists(r_path):
        os.mkdir(r_path)
    return r_path

# Segments data into different segments for testing
def segment_labels(true, predictions, predictions_proba, segment_size):
    # Find number of observations in the data
    segmented_true = []
    segmented_predictions = []
    segmented_predictions_proba = []

    for i in range(len(true)):
        subject_segmented_true = []
        subject_segmented_predictions = []
        subject_segmented_predictions_proba = []
        for j in range(0,len(true[i])-segment_size):
            # Majority voting for the separate segments
            subsequence_true = int(mode(true[i][j:(j+segment_size)]))
            subsequence_predictions = int(mode(predictions[i][j:(j+segment_size)]))
            subsequence_predictions_proba = round(np.mean(predictions_proba[i][j:(j+segment_size)]))
            subject_segmented_true.append(subsequence_true)
            subject_segmented_predictions.append(subsequence_predictions)
            subject_segmented_predictions_proba.append(subsequence_predictions_proba)
        segmented_true.append(subject_segmented_true)
        segmented_predictions.append(subject_segmented_predictions)
        segmented_predictions_proba.append(subject_segmented_predictions_proba)
    return segmented_true,segmented_predictions, segmented_predictions_proba

def store_parameters(test_name, new_test_values):
    results_path = f"Results\\{test_name}"
    file_name = os.path.join(f"Results\\{test_name}","parameters.json")
    parameters = []

    if not os.path.exists(results_path):
        os.mkdir(results_path)
    
    if os.path.exists(file_name):
        with open(file_name, 'r') as json_file:
            parameters = json.load(json_file)
        parameters['test_values'].extend(new_test_values)
    else: 
        parameters = {
            'test_values': new_test_values
        }
    
    with open(file_name,'w') as json_file:
        json.dump(parameters,json_file,indent=3)