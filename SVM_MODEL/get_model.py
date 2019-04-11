import numpy as np
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import GridSearchCV


def trainmodel(input_img, out_path='group_6.model', in_C=0.8, in_kernel='linear'):
    clf = SVC(C=in_C, kernel=in_kernel, probability=True)
    img_data = []
    img_tag = []
    for img in input_img:
        img_data.append(img[0])
        img_tag.append(img[1])
    clf.fit(img_data, img_tag)
    with open(out_path, 'wb') as f:
        pickle.dump(clf, f)


def to_model(train_image, train_label, out_path='model', adept=False):
    if not adept:
        clf = SVC(C=0.8, kernel='linear', probability=True)
        clf.fit(train_image, train_label)
        with open(out_path, 'wb') as f:
            pickle.dump(clf, f)
    else:
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.8, 0.9, 1, 10, 100]},
                            {'kernel': ['linear'], 'C': [0.5, 0.6, 0.7, 0.8, 0.9, 1, 10]}]
        scores = ['precision', 'recall']
        temp_clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='precision_macro')
        temp_clf.fit(train_image, train_label)
        print(temp_clf.best_params_)
        clf = SVC(C=temp_clf.best_estimator_['C'], kernel=temp_clf.best_estimator_['kernel'])
        clf.fit(train_image, train_label)
    return clf
