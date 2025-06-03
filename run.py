import machine_learning
import LDA
import hmm
import rnn
import l_predict
import feature_selection
import hyperparameter_optimization

def run_test(model,validation_set,test_name,features,labels,recording_ids,model_arcitechture,segment_parameters, multiclass):
    # Running machine learning on features
    train_data, train_labels, train_recording_ids, test_data, test_labels, test_recording_ids = machine_learning.split_data(features,labels,recording_ids,[1,4,20,21,27,31,38,43,45])
    
    # LDA
    if model.upper() == "LDA":
        if validation_set:
            LDA.run_LDA_train_and_validation(train_data,train_labels,train_recording_ids,test_name,segment_parameters, multiclass)
        else: 
            LDA.train_test(train_data, train_labels, train_recording_ids, test_data, test_labels, test_recording_ids,test_name,segment_parameters)
    # HMM
    elif model.upper() == "HMM":
        if validation_set:
            hmm.run_HMM_model_train_and_validation(data=train_data, labels=train_labels, recording_ids=train_recording_ids,test_name=test_name,model_arcitechture=model_arcitechture,segment_parameters=segment_parameters,multiclass=multiclass)
        else: 
            hmm.run_HMM_model_test(train_data, train_labels, train_recording_ids, test_data, test_labels, test_recording_ids,test_name,model_arcitechture,segment_parameters=segment_parameters)
    # RNN
    elif model.upper() == "RNN":
        rnn.train_test(train_data, train_labels, train_recording_ids, test_data, test_labels, test_recording_ids,test_name,segment_parameters)
    # LazyPredict
    elif model.upper() == "L_P":
        l_predict.train_test(train_data,train_labels,test_data,test_labels)
    # Feature selection LDA
    elif model.upper() == "SELECTION_LDA":
        feature_selection.feature_selection_LDA(train_data, train_labels,train_recording_ids)
    elif model.upper() == "SELECTION_HMM":
        feature_selection.feature_selection_HMM(train_data, train_labels, train_recording_ids, model_arcitechture)
    elif model.upper() == "OPTIMIZATION_HMM":
        hyperparameter_optimization.hyperparameter_optimization_HMM(train_data,train_labels,train_recording_ids)
    elif model.upper() == "OPTIMIZATION_LDA":
        hyperparameter_optimization.hyperparameter_optimization_LDA(train_data,train_labels,train_recording_ids)
            