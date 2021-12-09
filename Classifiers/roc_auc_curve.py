# -*- coding: utf-8 -*-
"""

Evaluation file that gives final ROC-AUC scores

@author: harro

"""

def plot_roc(classifier):
    """ Plotter for ROC AUC Curves, input should be a sklearn (like) clf"""
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.rcParams.update({'font.serif':'Times New Roman', 'font.size': 8})
    plt.rc('xtick', labelsize=8) 
    plt.rc('ytick', labelsize=8) 
    # Code taken from sklearn example: 
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(classifier, X[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.legend(loc="lower right", prop={'size':10})
    plt.show()
    pass

def plot_roc_nn(model, X, y):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import plot_roc_curve, auc
    
    from keras.wrappers.scikit_learn import KerasClassifier
    from neural_net import sklearn_compatible_model
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
    model = KerasClassifier(build_fn=sklearn_compatible_model,
                            epochs=10,
                            batch_size=32,
                            verbose=0)
    
    """ Plotter for ROC AUC Curves, input should be a sklearn (like) clf"""
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.rcParams.update({'font.serif':'Times New Roman', 'font.size': 8})
    plt.rc('xtick', labelsize=8) 
    plt.rc('ytick', labelsize=8) 
    
    clf = model.fit(X, y)
    nn_disp = plot_roc_curve(clf, X, y)
    nn_disp.plot()
    plt.show()
    
    # Code taken from sklearn example: 
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    for i, (train, test) in enumerate(cv.split(X, y)):
        clf = model.fit(X[train], y[train])
        print(clf)
        viz = plot_roc_curve(clf, X[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.legend(loc="lower right", prop={'size':10})
    plt.show()
    pass


def plot_pr_curve(classifier):
    """ Plots precision-recall curve for any sklearn (like) clf"""
    classifier.fit(X_train, y_train)
    y_score = classifier.decision_function(X_test)
    average_precision = average_precision_score(y_test, y_score)
    disp = plot_precision_recall_curve(classifier, X_test, y_test)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                       'AP={0:0.2f}'.format(average_precision))
    pass

# plot_pr_curve(rbf)
if __name__ == '__main__':
    import numpy as np
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    import matplotlib.pyplot as plt
    from sklearn.metrics import plot_roc_curve, auc
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import plot_precision_recall_curve
    
    # Import function to preprocess data
    import preprocessing_data
    X, y = preprocessing_data.preprocessing()
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=0.2,
                                                        random_state=42)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
    # Initialize classifier and fit on data
    rbf = SVC(kernel='rbf', class_weight='balanced', C=8, gamma=0.5) # C = 10 from gridsearch (see hyperparameter_tuning.py)
    linear = SVC(kernel='linear', class_weight='balanced', C=1, gamma=1)
    nb = GaussianNB(var_smoothing=1e-7)
    
    # Plot ROC-AUC curves
    plot_roc(nb)
    plot_roc(rbf)
    plot_roc(linear)