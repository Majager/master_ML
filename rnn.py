import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import datetime
import pickle
import machine_learning

class RNN:
    def __init__(self):
        self.model = Sequential([
            Input(shape=(46,1)),
            LSTM(46, return_sequences=True),
            Dropout(0.2),
            LSTM(20, return_sequences=False),
            Dropout(0.2),
            Dense(10, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, X, y, batch_size=32, epochs=20, validation_split=0.2):
        self.history = self.model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")
    
    def predict_proba(self, X):
        return (self.model.predict(X)).astype("float32")

def train_test(train_data, train_labels, train_recording_ids, test_data, test_labels, test_recording_ids,test_name,segment_parameters):
    classifier = RNN()
    train_data, train_labels = np.concatenate(train_data,axis=0), np.concatenate(train_labels,axis=0)
    classifier.fit(train_data, train_labels)

    predictions = []
    predictions_proba = []
    for idx in range(len(test_data)):
        predictions.append(classifier.predict(test_data[idx]))
        predictions_proba.append(classifier.predict_proba(test_data[idx]))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
    true_segmented, predictions_segmented, predictions_segmented_proba = machine_learning.segment_labels(test_labels,predictions,predictions_proba,segment_parameters[2])
        
    # Store results to pickle file
    r_path = machine_learning.store_results_filename(test_name,timestamp)
    full_path = os.path.join(r_path,f"test.pickle")
    with open(full_path,'wb') as handle:
        pickle.dump([true_segmented,predictions_segmented,predictions_segmented_proba,test_recording_ids,segment_parameters],handle,protocol=pickle.HIGHEST_PROTOCOL)
    machine_learning.store_parameters(test_name, ["test"])