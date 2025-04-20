from sklearn.feature_selection import RFECV, RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import hmm

def feature_selection_CV(estimator,features,labels):
    selector = RFECV(estimator,step=1,cv=3)
    train_data, train_labels = np.concatenate(features,axis=0), np.concatenate(labels,axis=0)
    selector = selector.fit(train_data,train_labels)
    return selector.ranking_, selector.support_

def feature_selection(estimator,features,labels):
    selector = RFE(estimator,step=2)
    train_data, train_labels = np.concatenate(features,axis=0), np.concatenate(labels,axis=0)
    selector = selector.fit(train_data,train_labels)
    return selector.ranking_, selector.support_

def feature_selection_LDA(features,labels):
    estimator = LinearDiscriminantAnalysis()
    features_LDA, features_LDA_include = feature_selection_CV(estimator,features,labels)
    print("Feature selection LDA")
    print(features_LDA)
    print(features_LDA_include)

def feature_selection_HMM(features,labels, model_arcitechture):
    estimator = hmm.HMM(model_arcitechture)
    features_HMM, features_HMM_include = feature_selection(estimator,features,labels)
    print("Feature selection HMM")
    print(features_HMM)
    print(features_HMM_include)
    