
"""
KNN implementation using NMSLIB for text classification
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import unique_labels

import scipy
from scipy.stats import mode

import numpy as np
import nmslib

class NMSlibKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=10, nmslib_method='hnsw', n_jobs=-1, metric="cosinesimil"):

        self.n_neighbors = n_neighbors
        self.nmslib_method = nmslib_method
        #self.cv: int = 0
        self.n_jobs: int = n_jobs
        self.metric = metric
        self.index_time_params = {}


    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, accept_sparse=True)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        if isinstance(X, scipy.sparse.csr.csr_matrix):
            self.index = nmslib.init(method=self.nmslib_method, space=f'{self.metric}_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
        elif isinstance(X, np.ndarray):
            self.index = nmslib.init(method=self.nmslib_method, space=self.metric)

        self.index.addDataPointBatch(X)
        #print('We have added %d data points' % nmslib.getDataPointQty(self.index))

        if self.nmslib_method == 'hnsw':
            self.index_time_params = {
            'M': 30, 'indexThreadQty': self.n_jobs, 'efConstruction': 100, 'post': 0}
            self.index.createIndex(self.index_time_params, print_progress=True)
        else:
            self.index.createIndex()

        # Return the classifier
        return self

    def predict(self, X):

        nearest_neighboor = np.zeros(
            (X.shape[0], self.n_neighbors), dtype=np.int64)
        for idx, x in enumerate(X):
            ids, distances = self.index.knnQuery(x, k=self.n_neighbors)
            nearest_neighboor[idx] = ids.tolist()

        modeResults = mode(self.y_[nearest_neighboor], axis=1)
        y_pred = modeResults.mode.ravel()

        return y_pred

    def predict_y_and_maxproba_for_X_train(self, X):

        nearest_neighboor = np.zeros(
            (X.shape[0], self.n_neighbors), dtype=np.int64)

        queryResults = self.index.knnQueryBatch(X, k=self.n_neighbors+1)

        for xidx in range(X.shape[0]):

            ids = queryResults[xidx][0].tolist()
            if xidx in ids:
                ids.remove(xidx)
            else:
                ids.pop()
            nearest_neighboor[xidx] = ids
        
        modeResults = mode(self.y_[nearest_neighboor], axis=1)
        y_pred = modeResults.mode.ravel()

        y_proba_of_pred = modeResults.count.ravel() / self.n_neighbors

        return y_pred, y_proba_of_pred