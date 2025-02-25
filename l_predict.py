from lazypredict.Supervised import LazyClassifier
import numpy as np

def train_test(train_data, train_labels, test_data, test_labels):
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    train_data, train_labels = np.concatenate(train_data,axis=0), np.concatenate(train_labels,axis=0)
    test_data, test_labels = np.concatenate(test_data,axis=0), np.concatenate(test_labels,axis=0)
    models, predictions = clf.fit(train_data, test_data, train_labels, test_labels)
    print(models)
