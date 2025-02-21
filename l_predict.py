from lazypredict.Supervised import LazyClassifier

def train_test(train_data, train_labels, test_data, test_labels):
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(train_data, test_data, train_labels, test_labels)
    print(models)