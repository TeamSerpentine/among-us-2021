# -*- coding: utf-8 -*-
"""
"Hyperparameter tuning for the automatic classification of player foles"

@author: harro
"""
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import pandas as pd

# Heatmap plotter taken from https://github.com/ML-course/master/tree/master/labs
def heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None,
            vmin=None, vmax=None, ax=None, fmt="%0.2f"):
    """
    Visualizes the results of a grid search with two hyperparameters as a heatmap.
    Attributes:
    values -- The test scores
    xlabel -- The name of hyperparameter 1
    ylabel -- The name of hyperparameter 2
    xticklabels -- The values of hyperparameter 1
    yticklabels -- The values of hyperparameter 2
    cmap -- The matplotlib color map
    vmin -- the minimum value
    vmax -- the maximum value
    ax -- The figure axes to plot on
    fmt -- formatting of the score values
    """
    if ax is None:
        ax = plt.gca()
    # plot the mean cross-validation scores
    img = ax.pcolor(values, cmap=cmap, vmin=None, vmax=None)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)
    
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12, labelrotation=90)

    for p, color, value in zip(img.get_paths(), img.get_facecolors(), img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, fmt % value, color=c, ha="center", va="center", size=10)
    return img

# Import preprocessed data
import preprocessing_data
X, y = preprocessing_data.preprocessing()

# Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.8, random_state=42)

# Initialize clfs
svm = SVC(kernel='linear', class_weight='balanced')
nb = GaussianNB()

# Setup parameters for gridsearch
resolution = 25
param_grid_svm = {'C': np.logspace(-6,6,resolution,base=2),
                  'gamma': np.logspace(-6,6,resolution,base=2)}
param_grid_nb = {'var_smoothing': np.logspace(0,-9, resolution),
                 'priors': ('None', [0.15,0.85])}

grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=3, n_jobs=-1, scoring='roc_auc').fit(X_train, y_train)
grid_search_nb = GridSearchCV(nb, param_grid_nb, cv=3, n_jobs=-1, scoring='roc_auc').fit(X_train, y_train)

# Plot gridsearch results for SVM
results = pd.DataFrame(grid_search_svm.cv_results_)
scores = np.array(results.mean_test_score).reshape(resolution, resolution)
plt.rcParams.update({'font.size': 18})
fig, axes = plt.subplots(1, 1, figsize=(13, 13))
heatmap(scores, xlabel='gamma', xticklabels=np.around(param_grid_svm['gamma'],4),
                      ylabel='C', yticklabels=np.around(param_grid_svm['C'],4), cmap="viridis", fmt="%.2f", ax=axes);

# Plot gridsearch results for NB
results = pd.DataFrame(grid_search_nb.cv_results_)
scores = np.array(results.mean_test_score).reshape(resolution, 2)
plt.rcParams.update({'font.size': 18})
fig, axes = plt.subplots(1, 1, figsize=(13, 13))
heatmap(scores, xlabel='priors', xticklabels=['None', '0.15, 0.85'],
                      ylabel='var_smoothing', yticklabels=np.around(param_grid_nb['var_smoothing'],4), cmap="viridis", fmt="%.2f", ax=axes);

