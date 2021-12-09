# -*- coding: utf-8 -*-
"""

@author: harro

"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, plot_roc_curve
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.utils import shuffle

# Import preprocessed data
import preprocessing_data
X, y = preprocessing_data.preprocessing()
# X, y, colors = shuffle(X,y,colors,random_state=42)
# # colors to ordinal int instead of str

# clean all 0 rows
X, y = pd.DataFrame(X), pd.DataFrame(y)
indices = X.index[X.eq(0).all(1)]
X.drop(indices, inplace=True)
y.drop(indices, inplace=True)
y = y.values.ravel()

# Setup pipeline
smote_pipe = make_pipeline(SMOTE(), SVC(kernel='rbf', C=5, gamma=0.5))

# Initialize classifier and fit on data
rbf = SVC(kernel='rbf', class_weight='balanced', C=5, gamma=0.5) # C = 10 from gridsearch (see hyperparameter_tuning.py)
linear = SVC(kernel='linear', class_weight='balanced', C=5, gamma=0.5)

clfs = [rbf,linear,smote_pipe]

for clf in clfs:
    print(clf, "Accuracy", np.mean(cross_val_score(clf, X, y, cv=5)))
    print(clf, "ROC AUC", np.mean(cross_val_score(clf, X, y, cv=5, scoring='roc_auc')))
    
    
