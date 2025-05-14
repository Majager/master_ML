from sklearn.model_selection import GridSearchCV, GroupKFold
import hmm
import numpy as np

class HMMWrapper(hmm.HMM):
    def __init__(self, n_components_nonmeal = 1, n_mix_nonmeal = 1, n_components_meal = 1, n_mix_meal = 1):
        model_arcitechture = {
            0: [n_components_nonmeal, n_mix_nonmeal], 
            1: [n_components_meal, n_mix_meal]
        }
        super().__init__(model_arcitechture)
        self.n_components_nonmeal = n_components_nonmeal
        self.n_mix_nonmeal = n_mix_nonmeal
        self.n_components_meal = n_components_meal
        self.n_mix_meal = n_mix_meal

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
        'n_components_nonmeal' : [1, 3, 5, 7, 9],
        'n_mix_nonmeal' : [1, 3, 5, 7, 9],
        'n_components_meal' : [1, 3, 5, 7, 9],
        'n_mix_meal' : [1, 3, 5, 7, 9]
    }
    search = GridSearchCV(estimator,param_grid=param_dist,scoring=hmm.custom_score,n_jobs=-1,cv=group_kfold,verbose=3)
    search = search.fit(train_data,train_labels,groups)

    print("Best parameters:", search.best_params_)
    print("Best score:", search.best_score_)
