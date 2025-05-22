from sklearn.model_selection import GridSearchCV, GroupKFold
import hmm
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class HMMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components_nonmeal = 1, n_mix_nonmeal = 1, n_components_meal = 1, n_mix_meal = 1):
        self.n_components_nonmeal = n_components_nonmeal
        self.n_mix_nonmeal = n_mix_nonmeal
        self.n_components_meal = n_components_meal
        self.n_mix_meal = n_mix_meal
        self.model = None
    
    def fit(self, X, y):
        model_arcitechture = {0:[self.n_components_nonmeal, self.n_mix_nonmeal], 1:[self.n_components_meal, self.n_mix_meal]}
        self.model = hmm.HMM(model_arcitechture)
        self.model.fit(X,y)
        return self
    
    def score(self,X,y):
        return self.model.score(X,y)

def hyperparameter_optimization_HMM(features,labels,recording_ids):
    train_data, train_labels, train_lengths = hmm.prepare_test_data(features,labels)
    estimator = HMMWrapper()
    group_kfold = GroupKFold(n_splits=5)
    groups = np.zeros(train_data.shape[0])
    iterable = 0
    for group_idx, length in enumerate(train_lengths):
        for groups_vector_idx in range(iterable,iterable+length):
            groups[groups_vector_idx] = group_idx
        iterable += length
    param_dist = {
        'n_components_nonmeal' : [1,3,5,7,9],
        'n_mix_nonmeal' : [1,3,5,7,9],
        'n_components_meal' : [1,3,5,7,9],
        'n_mix_meal' : [1,3,5,7,9]
    }
    search = GridSearchCV(estimator,param_grid=param_dist,scoring=hmm.custom_score,n_jobs=-1,cv=group_kfold,verbose=3)
    search = search.fit(train_data,train_labels,groups=groups)

    print("Best parameters:", search.best_params_)
    print("Best score:", search.best_score_)

    cv_results_df = pd.DataFrame(search.cv_results_)
    cv_results_df.to_csv(f"Results\\cv_results_14_features_2.csv", index=False)


