import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import datetime
import pickle
import machine_learning
import keras_tuner as kt
from sklearn.model_selection import KFold

class RNN:
    def __init__(self, hp=None):
        self.model = Sequential([
            Input(shape=(43,1)),
            LSTM(
                units = hp.Int('lstm_1',min_value=16, max_value=128,step=16), 
                return_sequences=True
            ),
            Dropout(hp.Float('dropout_1',min_value=0.05,max_value=0.45, step=0.1)),
            LSTM(
                units = hp.Int('lstm_2',min_value=16, max_value=64,step=8), 
                return_sequences=False
            ),
            Dropout(hp.Float('dropout_2',min_value=0.05,max_value=0.45, step=0.1)),
            Dense(
                units = hp.Int('dense_units',min_value = 8, max_value = 64, step = 8),
                activation='relu'
            ),
            Dense(1, activation='sigmoid')  # Binary classification
        ])

        self.model.compile(optimizer='adam',
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )

    def train(self, X, y, batch_size=32, epochs=20, validation_split=0.2):
        self.history = self.model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")
    
def build_rnn(hp):
    return RNN(hp).model

def run_rnn_hyperparameters_search(features,labels,recording_ids,test_name,segment_parameters):
    # Cross-validation loop to be able to average over all folds
    kfold = KFold(n_splits=len(features), shuffle=True)
    for fold, (train_idx, validation_idx) in enumerate(kfold.split(features)):
        # Split data
        train_data, train_labels, _, validation_data, validation_labels, validation_recording_ids = machine_learning.split_data(features,labels,recording_ids,validation_idx)
        train_data, train_labels = np.concatenate(train_data,axis=0), np.concatenate(train_labels,axis=0)
        validation_data, validation_labels = np.concatenate(validation_data,axis=0), np.concatenate(validation_labels,axis=0)

        tuner = kt.BayesianOptimization(
            build_rnn,
            objective = 'val_accuracy',
            max_trials = 50 # Hyperparameter combinations
        )

        tuner.search(train_data, train_labels, epochs=20,validation_data=(validation_data,validation_labels))
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hps)