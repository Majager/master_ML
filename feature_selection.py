from sklearn.feature_selection import RFECV, RFE, mutual_info_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from mlxtend.feature_selection import SequentialFeatureSelector
import numpy as np
import hmm
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import KFold
import machine_learning
import datetime
import time
import os
import hmm

def RFE_selection(estimator,train_data,train_labels):
    selector = RFE(estimator,step=1,n_features_to_select=1)
    selector = selector.fit(train_data,train_labels)
    return selector.ranking_

def feature_selection_LDA_algorithms(features,labels):
    # Merge features as 1 vector
    train_data, train_labels = np.concatenate(features,axis=0), np.concatenate(labels,axis=0)

    # Mutual information as a filter method
    print("Feature selection with mutual information")
    importance_mutual_information = mutual_info_classif(train_data,train_labels)
    importance_mutual_information_ranks = convert_mutual_information(importance_mutual_information)
    print(importance_mutual_information_ranks)

    # RFE as an embedded method
    estimator_RFE = LinearDiscriminantAnalysis()
    importance_RFE = RFE_selection(estimator_RFE,train_data,train_labels)
    print("Feature selection LDA with RFE")
    print(importance_RFE)

    # # Sequential Feature Selector as a wrapper method
    print("Feature selection LDA with Forward selection")
    importance_sfs = np.zeros(train_data.shape[1],dtype=int)
    estimator_sfs = LinearDiscriminantAnalysis()
    selector = SequentialFeatureSelector(estimator_sfs,k_features=len(importance_sfs)-1,cv=5,n_jobs=-1,scoring='f1_macro')
    selector = selector.fit(train_data,train_labels)
    for i in range(1,len(selector.subsets_)+1):
        indices = selector.subsets_[i]['feature_idx']
        for j in indices:
            if importance_sfs[j] == 0:
                importance_sfs[j] = i
    for i in range(len(importance_sfs)):
        if importance_sfs[i] == 0:
            importance_sfs[i] = len(importance_sfs)
    print(importance_sfs)

    # Store features for later use
    with open(f'feature_selection_LDA.pickle', 'wb') as handle:
        pickle.dump([importance_mutual_information_ranks,importance_RFE,importance_sfs],handle, protocol=pickle.HIGHEST_PROTOCOL)

def feature_selection_HMM_algorithms(features,labels, model_arcitechture):
    train_data, train_labels, train_lengths = hmm.prepare_test_data(features,labels)

    # Mutual information as a filter method
    print("Feature selection with mutual information")
    importance_mutual_information = mutual_info_classif(train_data,train_labels)
    importance_mutual_information_ranks = convert_mutual_information(importance_mutual_information)
    print(importance_mutual_information_ranks)

    # Sequential Feature Selection
    importance_sfs = np.zeros(train_data.shape[1],dtype=int)
    estimator = hmm.HMM(model_arcitechture)
    group_kfold = GroupKFold(n_splits=5)
    groups = np.zeros(train_data.shape[0])
    iterable = 0
    for group_idx, length in enumerate(train_lengths):
        for groups_vector_idx in range(iterable,iterable+length):
            groups[groups_vector_idx] = group_idx
        iterable += length
    selector = SequentialFeatureSelector(estimator,k_features=len(importance_sfs)-1,cv=group_kfold,n_jobs=-1,scoring = hmm.custom_score)
    selector = selector.fit(train_data,train_labels,groups=groups)
    for i in range(1,len(selector.subsets_)+1):
        indices = selector.subsets_[i]['feature_idx']
        for j in indices:
            if importance_sfs[j] == 0:
                importance_sfs[j] = i
    for i in range(len(importance_sfs)):
        if importance_sfs[i] == 0:
            importance_sfs[i] = len(importance_sfs)
    print(importance_sfs)

    # Store features for later use
    with open(f'feature_selection_HMM.pickle', 'wb') as handle:
        pickle.dump([importance_mutual_information_ranks, importance_sfs],handle, protocol=pickle.HIGHEST_PROTOCOL)

def extract_selected_features(data,indices):
    num_subjects = len(data)
    features = np.empty(num_subjects,dtype=object)
    indices = np.array(indices)
    for recording_idx, recording in enumerate(data):
        recording_data = []
        for segment_idx in range(len(recording)):
            segment = np.asarray(recording[segment_idx])
            selected_features = segment[indices]
            recording_data.append(selected_features)
        recording_data = np.array(recording_data, dtype=np.float32)
        features[recording_idx] = recording_data
    return features

def feature_selection_LDA_base(importance,features,labels,recording_ids,test_name):
    max_value = np.max(importance)
    values = list(range(1,max_value+1))
    indices_list = [np.where(np.isin(importance,values[:i]))[0] for i in range(1,len(values)+1)]

    for indices in indices_list:
        selected_features = extract_selected_features(features,indices)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Cross-validation loop to be able to average over all folds
        kfold = KFold(n_splits=10, shuffle=True)
        for fold, (train_idx, validation_idx) in enumerate(kfold.split(selected_features)):
            # Split data
            train_data, train_labels, _, validation_data, validation_labels, validation_recording_ids = machine_learning.split_data(selected_features,labels,recording_ids,validation_idx)
            # Train LDA
            classifier = LinearDiscriminantAnalysis()
            train_data, train_labels = np.concatenate(train_data,axis=0), np.concatenate(train_labels,axis=0)
            classifier.fit(train_data, train_labels)
            predictions = []
            predictions_proba = []
            for idx in range(len(validation_data)):
                predictions.append(classifier.predict(validation_data[idx]))
                predictions_proba.append(classifier.predict_proba(validation_data[idx]))
            r_path = machine_learning.store_results_filename(test_name,timestamp)
            full_path = os.path.join(r_path,f"fold{fold+1}.pickle")
            with open(full_path,'wb') as handle:
                pickle.dump([validation_labels,predictions,predictions_proba,validation_recording_ids,[1,0.5,1]],handle,protocol=pickle.HIGHEST_PROTOCOL)
        time.sleep(1)
    machine_learning.store_parameters(test_name, values)

def convert_mutual_information(mutual_information):
    mutual_information_copy = np.copy(mutual_information)
    ranks = np.zeros(len(mutual_information_copy),dtype=int)
    rank_given = np.min(mutual_information_copy)-1
    for rank in range(1,len(ranks)+1):
        current_max, max_idx = mutual_information_copy[0], 0
        for idx, value in enumerate(mutual_information_copy):
            if value>current_max:
                current_max, max_idx = value, idx
        ranks[max_idx] = rank
        mutual_information_copy[max_idx] = rank_given
    return ranks

def feature_selection_LDA(features,labels,recording_ids):
    #feature_selection_LDA_algorithms(features,labels)

    importance_mutual_information, importance_RFE, importance_sfs = [],[],[]
    # Extract features from previous calculations
    with open(f'feature_selection_LDA.pickle', 'rb') as handle:
        importance_mutual_information, importance_RFE, importance_sfs = pickle.load(handle)

    for run in range (1,11):
        feature_selection_LDA_base(importance_mutual_information,features,labels,recording_ids,f"mutual_information_LDA\\run_{run}")

        feature_selection_LDA_base(importance_RFE,features,labels,recording_ids,f"RFE_LDA\\run_{run}")

        feature_selection_LDA_base(importance_sfs,features,labels,recording_ids,f"sfs_LDA\\run_{run}")

def feature_selection_HMM_base(importance,features,labels,recording_ids,test_name,model_arcitechture):
    max_value = np.max(importance)
    values = list(range(1,max_value+1))
    indices_list = [np.where(np.isin(importance,values[:i]))[0] for i in range(1,len(values)+1)]

    for indices in indices_list:
        selected_features = extract_selected_features(features,indices)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Cross-validation loop to be able to average over all folds
        kfold = KFold(n_splits=5, shuffle=True)
        for fold, (train_idx, validation_idx) in enumerate(kfold.split(selected_features)):
            # Split data
            train_data, train_labels, _, validation_data, validation_labels, validation_recording_ids = machine_learning.split_data(selected_features,labels,recording_ids,validation_idx)
            # Train HMM
            hmm_model = hmm.HMM(model_arcitechture)
            hmm_model = hmm.train_manager(hmm_model,train_data,train_labels)

            # Test HMM model based on the validation fold of this iteration
            true, predictions, predictions_proba = hmm.test_manager(hmm=hmm_model,data=validation_data,labels=validation_labels,n_segments=70)

            r_path = machine_learning.store_results_filename(test_name,timestamp)
            full_path = os.path.join(r_path,f"fold{fold+1}.pickle")
            with open(full_path,'wb') as handle:
                pickle.dump([true,predictions,predictions_proba,validation_recording_ids,[1,0.5,70]],handle,protocol=pickle.HIGHEST_PROTOCOL)
        time.sleep(1)
    machine_learning.store_parameters(test_name, values)

def feature_selection_HMM(features,labels,recording_ids,model_arcitechture):
    # feature_selection_HMM_algorithms(features,labels,model_arcitechture)

    importance_mutual_information, importance_sfs = [],[]
    # Extract features from previous calculations
    with open(f'feature_selection_HMM.pickle', 'rb') as handle:
        importance_mutual_information, importance_sfs = pickle.load(handle)

    run = 3
    # feature_selection_HMM_base(importance_mutual_information,features,labels,recording_ids,f"mutual_information_HMM\\run_{run}",model_arcitechture)
    feature_selection_HMM_base(importance_sfs,features,labels,recording_ids,f"sfs_HMM\\run_{run}",model_arcitechture)

def feature_selection_extraction(features,model):
    if model == "OPTIMIZATION_HMM" or model == "PROBABILITY_HMM":
        model = "HMM"
    elif model == "OPTIMIZATION_LDA" or model == "PROBABILITY_LDA":
        model = "LDA"

    selected_features_idx = []
    with open(f'selected_features_{model}.pickle','rb') as handle:
        selected_features_idx = pickle.load(handle)[0]
    features = extract_selected_features(features,selected_features_idx)
    return features