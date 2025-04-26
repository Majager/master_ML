from sklearn.feature_selection import RFECV, RFE, SequentialFeatureSelector, mutual_info_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import hmm
import matplotlib.pyplot as plt
import pickle

def RFE_CV(estimator,train_data,train_labels):
    selector = RFECV(estimator,step=1,cv=5)
    selector = selector.fit(train_data,train_labels)
    return selector.ranking_, selector.cv_results_

def RFE_selection(estimator,train_data,train_labels):
    selector = RFE(estimator,step=1,n_features_to_select=1)
    selector = selector.fit(train_data,train_labels)
    return selector.ranking_

def SequentialFeatureSelection(estimator, train_data,train_labels):
    selector = SequentialFeatureSelector(estimator,cv=5)
    selector = selector.fit(train_data,train_labels)
    return selector.support_

def feature_selection_LDA(features,labels):
    # Merge features as 1 vector
    train_data, train_labels = np.concatenate(features,axis=0), np.concatenate(labels,axis=0)
    
    # Mutual information as a filter method
    print("Feature selection with mutual information")
    importance_mutual_information = mutual_info_classif(train_data,train_labels)
    print(importance_mutual_information)

    # RFE as an embedded method
    estimator_RFE = LinearDiscriminantAnalysis()
    importance_RFE = RFE_selection(estimator_RFE,train_data,train_labels)
    print("Feature selection LDA with RFE")
    print(importance_RFE)

    # RFE CV 
    # estimator_RFECV = LinearDiscriminantAnalysis()
    # importance_RFECV, importance_RFECV_results = RFE_CV(estimator_RFECV,train_data,train_labels)
    # print("Feature selection LDA with RFE CV")
    # print(importance_RFECV)
    # plt.figure()
    # plt.plot(importance_RFECV_results['n_features'],importance_RFECV_results['mean_test_score'])
    # plt.fill_between(importance_RFECV_results['n_features'],importance_RFECV_results['mean_test_score']-importance_RFECV_results['std_test_score'],importance_RFECV_results['mean_test_score']+importance_RFECV_results['std_test_score'],alpha=0.2)
    # plt.xlabel('Number of selected features')
    # plt.ylabel('True')
    # plt.title('Performance of LDA')
    # plt.grid(True)
    # plt.show()

    # Sequential Feature Selector as a wrapper method
    print("Feature selection LDA with Forward selection")
    importance_sfs = np.zeros(train_data.shape[1],dtype=int)
    estimator_sfs = LinearDiscriminantAnalysis()
    for i in range(len(importance_sfs)-1):
        selector = SequentialFeatureSelector(estimator_sfs,cv=3,n_features_to_select=i+1)
        selector = selector.fit(train_data,train_labels)
        indices = selector.get_support(indices=True)
        for j in indices:
            if importance_sfs[j]==0:
               importance_sfs[j] = i+1
    print(importance_sfs) 

    # Store features for later use
    with open(f'LDA_feature_selection.pickle', 'wb') as handle:
        pickle.dump([importance_mutual_information,importance_RFE,importance_sfs],handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def feature_selection_HMM(features,labels, model_arcitechture):
    train_data, train_labels = np.concatenate(features,axis=0), np.concatenate(labels,axis=0)
    estimator = hmm.HMM(model_arcitechture)
    features_HMM = RFE_selection(estimator,train_data,train_labels)
    print("Feature selection HMM")
    print(features_HMM)
    