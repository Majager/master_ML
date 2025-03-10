from sklearn.feature_selection import RFECV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

def feature_selection(estimator,features,labels):
    selector = RFECV(estimator,step=1,cv=5)

    train_data, train_labels = np.concatenate(features,axis=0), np.concatenate(labels,axis=0)
    selector = selector.fit(train_data,train_labels)
    
    return selector.ranking_, selector.support_

def feature_selection_LDA(features,labels):
    estimator = LinearDiscriminantAnalysis()
    features_LDA, features_LDA_include = feature_selection(estimator,features,labels)
    print(features_LDA)
    print(features_LDA_include)
