#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import KFold
import sklearn.neighbors
import pandas as pd
#%%
# load data
data = load_iris()

# extract features, feature names and targets
X = data['data']
y = data['target']
feature_names = data['feature_names']

# Make dataframe from features and targets
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
# %%
# Define variables
N_FOLDS = 5 # number of folds in cross validation
MAX_K = 25 # maximum number of neighbours in KNN
MIN_K = 2
N_CV = 100 # number of CV runs
ksize = np.arange(2, MAX_K+1)
nmodels = len(ksize)
cvErr = np.zeros((nmodels, N_CV))
kf_state = np.random.randint(0, N_CV*10, N_CV)
metrics = ['minkowski', 'euclidean']
#%%
# Evaluate mean CV accuracy after a number of CV runs 
# for different KNN classifiers
for metric in metrics:

    for j in range(N_CV):

        kf = KFold(n_splits=N_FOLDS, 
                   shuffle=True, 
                   random_state=kf_state[j])

        for i in range(nmodels):

                knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=ksize[i])
                acc = 0

                for train_index, val_index in kf.split(X):

                    knn.fit(X[train_index], y[train_index])
                    y_pred = knn.predict(X[val_index])
                    acc += accuracy_score(y_pred, y[val_index])

                acc /= N_FOLDS # compute mean accuracy for the model
                cvErr[i][j] = acc

#%%
    cvAvg = np.mean(cvErr, axis=1)
    cvStd = np.std(cvErr, axis=1)
# %%
    plt.errorbar(ksize, cvAvg, yerr=cvStd, marker='x', capsize=3)
    plt.grid(True)
    plt.xlabel('K')
    plt.ylabel('Mean CV accuracy')
    plt.title('Distance metric - {}'.format(metric))
    plt.show()
# %%

# %%
