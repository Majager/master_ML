import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import datetime
import pickle

class RNN:
    def __init__(self):
        self.model = Sequential([
            Input(shape=(39,1)),
            LSTM(39, return_sequences=True),
            Dropout(0.2),
            LSTM(20, return_sequences=False),
            Dropout(0.2),
            Dense(10, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X, y, batch_size=32, epochs=20, validation_split=0.2):
        self.history = self.model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")

def split_data_to_train_and_test(data,labels,train_indices,test_indices,recording_ids):
    test_data = [data[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    test_recording_ids = [recording_ids[i] for i in test_indices]

    # The rest of the data will be in the training set
    train_data = [data[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_recording_ids = [recording_ids[i] for i in train_indices]
    return train_data, train_labels, train_recording_ids, test_data, test_labels, test_recording_ids

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

def train_test(features,labels,recording_ids,test_name,segment_parameters):
    classifier = RNN()

    train_data, train_labels, train_recording_ids, test_data, test_labels, test_recording_ids = split_data_to_train_and_test(features,labels,[1,2,3],[0],recording_ids)
    
    train_data, train_labels = np.concatenate(train_data,axis=0), np.concatenate(train_labels,axis=0)
    classifier.train(train_data, train_labels)

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
