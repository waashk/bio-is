
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import time
import gzip
import pickle

base_estimators = {
    'knn': KNeighborsClassifier(),
}

default_params = {
    'knn': 	{'n_neighbors': 30, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 30,  # 30,
             'p': 2, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': -1},
}

default_tuning_params = {
    'knn': [{'n_neighbors': [10, 40, 70, 100]}],
}

class KNNTraditionalClassifier(BaseEstimator):

    def __init__(self, n_neighbors: int = 0, 
                        cv: int = 0,
                        n_jobs: int = -1,
                        metric: str = "minkowski"):

        self.n_neighbors = n_neighbors
        self.cv = cv
        self.n_jobs = n_jobs
        self.metric = metric 
        
        self.estimator = base_estimators['knn']
        self.params = default_params['knn'].copy()
        

        if self.n_neighbors:
            self.params['n_neighbors'] = self.n_neighbors
        
        if self.metric:
            self.params['metric'] = self.metric

        self.estimator.set_params(**self.params)

        self.grid_time = 0
        self.train_time = 0
        self.test_time = 0

        #self.GridSearchCVvalues = self.args['GridSearchCVvalues']

        #print(self.estimator)

    def gridSearch(self, X, y):

        if X.shape[0] < 120:
            print("Adjusting KNN Default Tunning parameters")

            default_tuning_params['knn'][0]['n_neighbors'] = sorted(
                list(set(map(int, np.linspace(1, 0.75*X.shape[0], 5)))))
            print(X.shape)
            

        #Possibilitando executar o cv para datasets ext pequenos
        #counter = Counter(y)
        #mininum = min(counter, key=counter.get)
        #if counter[mininum] < self.cv and self.args['dataset'] not in ['webkb', '20ng', 'acm', 'reut', 'reut90']:
        #    print(f"Adjusting CV value to {counter[mininum]}")
        #    self.cv = counter[mininum]
        #    #print(self.args)

        t_init = time.time()
        gs = GridSearchCV(self.estimator, default_tuning_params['knn'],
        #gs = GridSearchCV(self.estimator, [{'n_neighbors': [3, 5, 10]}],
                          n_jobs=self.n_jobs,
                          # refit=False,
                          cv=self.cv,
                          #verbose=1,
                          verbose=0,
                          scoring='f1_macro')

        gs.fit(X, y)
        #print(gs.best_score_, gs.best_params_)

        self.estimator.set_params(**gs.best_params_)
        self.grid_time = time.time() - t_init

        #print(self.grid_time)
        self.params['n_neighbors'] = gs.best_params_['n_neighbors']


    def fit(self, X, y=None):

        # self.args['best_param_class'].append(gs.best_params_)

        #print(self.estimator)
        #self.estimator = clone(self.estimator)

        # fit and predict
        #print('Fitting')
        t_init = time.time()
        self.estimator.fit(X, y)
        self.train_time = time.time() - t_init
        #print(f"{self.train_time} seconds to fit")

        t_init = time.time()
        # calibrator pro predict proba
        self.calibrator = CalibratedClassifierCV(self.estimator, cv='prefit')
        self.calibrator.fit(X, y)
        #print(f"{time.time() - t_init} seconds to calibrate")

        return self

    def predict(self, X, y=None):

        #print('Predicting')
        t_init = time.time()
        self.y_pred = self.estimator.predict(X)
        self.test_time = time.time() - t_init
        # self.args['inst_time']['time_test_class'].append(t)
        # self.args['y_pred'].append(self.y_pred.tolist())
        return self.y_pred

    def predict_proba(self, X, y=None):
        # if self.args['name_class'] == 'lsvm':
        #	y_margins = self.estimator.decision_function(X)
        #	return (y_margins - y_margins.min()) / (y_margins.max() - y_margins.min())
        # else:
        #return self.estimator.predict_proba(X)
        t_init = time.time()
        result = self.calibrator.predict_proba(X) 
        #print(f"{time.time() - t_init} seconds to predict")
        return result






def generateExactKNN(X, y, cvSplit, doGridSearch, n_neighbors):

        #skf = StratifiedKFold(n_splits=5)

        # Info do Knn
        #info = {#"dataset": self.args.dataset,
        #        "name_class": "knn", 'n_jobs': -1, 'cv': 0,
        #    }

        if doGridSearch:
            #info['cv'] = 5
            classifier = KNNTraditionalClassifier(n_neighbors=n_neighbors,
                                                  cv=5,
                                                  n_jobs=-1)
            classifier.gridSearch(X, y)
            n_neighbors = classifier.params['n_neighbors']

        y_pred = np.zeros(y.size) - 1.0
        max_proba_by_instance = np.zeros(y.size)
        macro_val_list = []

        #for train_index, test_index in skf.split(X, y):
        for train_index, test_index in cvSplit:

            #X_train = 

            classifier = KNNTraditionalClassifier(n_neighbors=n_neighbors)
            classifier.fit(X[train_index], y[train_index])

            #print(classifier.estimator.score(X[test_index], y[test_index]))
            #print(classifier.estimator.predict_proba(X[test_index]))
            #print(Counter(y))

            # Prediz probabilidades para instancias no treino atual - test partition
            proba = classifier.predict_proba(X[test_index])
            y_pred[test_index] = proba.argmax(axis=1)

            macro = f1_score(y_true=y[test_index],
                             y_pred=y_pred[test_index], average='macro')

            macro_val_list.append(macro)

            max_proba_by_instance[test_index] = proba.max(axis=1)

        #print(str(macro_val_list))

        return y_pred, macro_val_list, max_proba_by_instance, n_neighbors