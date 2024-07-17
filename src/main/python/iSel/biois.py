import numpy as np
import copy
from scipy import stats

from sklearn.utils.validation import check_X_y
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import StratifiedKFold

from src.main.python.iSel.base import InstanceSelectionMixin

class BIOIS(InstanceSelectionMixin):
    """ Bi-objective instance selection framework (biO-IS)
    Description:
    ==========

    We describe here the main contribution of [1]: the proposal of bio-IS - a novel bi-objective IS framework aimed at simultaneously removing redundant and noisy instances from the training set.

    Our extended framework encompasses three main components: a weak classifier, a redundancy-based approach, and an entropy-based approach. To address the first objective of redundancy removal, we depart from our original solution proposed in [2], in which we replace the KNN component (in our original work) with a calibrated weak classifier learnt via logistic regression. An in-depth comparative analysis of several possibilities of the weak classifiers, including Decision Tree (DT), Logistic Regression (LR), XGBoost, LightGBM (LGBM), and Linear SVM, proved LR the best option in terms of a trade-off effectiveness-calibration-cost.

    To address the second objective of noise removal, we propose a new step based on the entropy and a novel iterative process to estimate near-optimum reduction rates. Considering the instances wrongly predicted by the weak classifier, the main objective is to assign a probability to each of them being removed from the training set based on the probability of the instance being noise. For this purpose, we propose a new entropy-based criterion. The rationale behind this new step is rooted in the observation that the entropy of the posterior probabilities correlates negatively with the confidence of the classifier. In other words, low entropy occurs when the classifier assigns an instance with high confidence to a wrong class, while high entropy occurs when the classifier is uncertain among several classes. Therefore, instances that are (i) incorrectly classified and that (ii) have low entropy should be more likely to be removed. Accordingly, we assign a higher removal probability (i.e., we consider an instance to be ``noisy'') by considering the inverse of the entropy of the prediction. Accordingly, the proposed biO-IS framework provides a comprehensive solution to address both redundancy and noise removal simultaneously.

    Parameters:
    ===========

    beta : float, default=0.0
        Beta reduction rate. It is only significant when betaMode == 'prefixed'

    theta : float, default=0.0
        theta reduction rate. It is only significant when betaMode == 'prefixed'

    maxreduction : float, default=1.0
        Maximum reduction rate considered when betaMode == 'iterative'


    Attributes:
    ==========

    mask : ndarray of shape
        Binary array indicating the selected instances.

    X_ : csr matrix
        Instances in the reduced training set.
    
    y_ : ndarray of shape
        Labels in the reduced training set.

    sample_indices_: ndarray of shape (q Instances in the reduced training set)
        Indices of the selected samples.

    reduction_ : float
        Reduction is as defined R = (|T| - |S|)/|T|, where |T| is the original training set, |S| is the solution set containing the selected instances by the IS method.

    classes_ : int 
        The unique classes labels.

    Ref.
    ====

    (UNDER-REVIEW)
    [1] Washington Cunha, Alejandro Moreo, Andrea Esuli, Fabrizio Sebastiani, Leonardo Rocha, and Marcos A. Gonçalves. A Noise-Oriented and Redundancy-Aware Instance Selection Framework. In the ACM Transactions on Information Systems (TOIS) journal.

    [2] Washington Cunha, Celso França, Guilherme Fonseca, Leonardo Rocha, and Marcos A. Gonçalves. An Effective, Efficient, and Scalable Confidence-based Instance Selection Framework for Transformer-based Text Classification. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR'23, New York, NY, USA, 2023. Association for Computing Machinery.
        
    Example
    =======

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from src.main.python.iSel import biois
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> selector = biois.BIOIS()
    >>> idx = selector.sample_indices_
    >>> X_train_selected, y_train_selected =  X_train[idx], y_train[idx]
    >>> print('Resampled dataset shape %s' % Counter(y_train_selected))
    Resampled dataset shape Counter({1: 36, 0: 14})
    """

    def __init__(self, beta = 0.0,
                       theta = 0.0,
                       maxreduction = 1.0):
        
        self.beta = beta
        self.theta = theta 
        self.maxreduction = maxreduction
        
        self.sample_indices_ = []

        
    def fitting_alpha(self, X, y):
        print('fitting_alpha_by_lr_default')
        # Setting the approximated KNN solution
        #classifier = NMSlibKNNClassifier(n_neighbors=10, n_jobs=10)
        #classifier.fit(X, y)

        nrows = X.shape[0]
        self.classes_ = unique_labels(y)
        ncolumns = len(self.classes_)
        probaEveryone = np.zeros((nrows,ncolumns))

        #sss = StratifiedShuffleSplit(n_splits=5, train_size=.8, random_state=0)
        sss = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        splits = []
        for train_index, val_index in sss.split(X, y):
            splits.append((train_index, val_index))
        
        for (train_index, val_index) in splits:
            X_train, y_train = X[train_index], y[train_index]
            X_val, y_val = X[val_index], y[val_index]

            classifier = LogisticRegression(C=1.0,solver='warn',multi_class='warn',n_jobs=-1)
            print(classifier)
            classifier.fit(X_train, y_train)

            probas = classifier.predict_proba(X_val)
            columns_diff = list(sorted(list(set(y)-set(y_train))))
            if columns_diff:
                probas = self.fix_proba_columns_if_necessary(probas, columns_diff, max(y_train))

            #probaEveryone[val_index] = classifier.predict_proba(X_val)
            probaEveryone[val_index] = copy.copy(probas)

        pred = np.argmax(probaEveryone, axis=1)
        print(f"Micro: {f1_score(y, pred,average='micro')}")
        print(f"Macro: {f1_score(y, pred,average='macro')}")

        # Predicting the probabilities using the approximated KNN solution
        #pred, probaEveryone = classifier.predict_y_and_maxproba_for_X_train(X)
        y_proba_of_pred = np.array([probaEveryone[l][pred[l]] for l in range(X.shape[0])])
        #print(pred)
        self._probaEveryone = copy.copy(probaEveryone)
        self._pred = copy.copy(pred)
        self._y_proba_of_pred = copy.copy(y_proba_of_pred)

        if f1_score(y, pred,average='micro') < self.beta:
            #raise ValueError("ERROR. LR accuracy < beta")
            print("ERROR. LR accuracy < beta")

        # Setting the removal probability of wrong predicted instances as zero
        correctPredictedProba = copy.copy(y_proba_of_pred)
        correctPredictedProba[pred != y] = 0.
        # Normalazing the results to reach the the alpha distribution
        correctPredictedProba = correctPredictedProba / np.sum(correctPredictedProba)

        #alpha = correctPredictedProba
        #print(len(y),sum((pred != y) == (correctPredictedProba == .0)))
        #print(sum(pred != y))
        #print(sum(correctPredictedProba == .0))
        return correctPredictedProba
    
    def identifyNoiseByLowerNNEntropy(self, X, y):
        print("identifyNoiseByLowerNNEntropy")
        wrongpredictedIdx = y != self._pred
       
        nnentropy = [stats.entropy(_) for _ in self._probaEveryone[wrongpredictedIdx]]
        nnentropy = np.array(nnentropy)

        nnentropy = (nnentropy-nnentropy.min())/(nnentropy.max()-nnentropy.min())
        nnentropy = 1. - nnentropy

        nnentropy /= nnentropy.sum()

        proba_toremove = np.zeros(X.shape[0])
        proba_toremove[wrongpredictedIdx] = nnentropy

        nwrong = sum(wrongpredictedIdx)
        ntoremove = int(self.theta * nwrong)

        idx_choice_to_remove = np.random.choice(a=list(range(X.shape[0])),
                                                size=ntoremove,
                                                replace=False,
                                                p=proba_toremove)

        return idx_choice_to_remove

    

    def select_end(self, alpha, beta):

        # Choosing the instances to be removed based on either alpha distibution and beta rate.
        n_training_samples = len(alpha)
        n_samples_to_remove = int(n_training_samples * beta)
        
        idx_choice_to_remove = np.random.choice(a=list(range(n_training_samples)),
                                                size=n_samples_to_remove,
                                                replace=False,
                                                p=alpha)
        
        return idx_choice_to_remove
   
    def fix_proba_columns_if_necessary(self, probas, columns_diff,max_y_train):

        #probas = copy.copy(classifier.predict_proba(X_val))
        #columns_diff = set(y)-set(y_train)
        n_instances = probas.shape[0]

        if columns_diff:
            for c in columns_diff:
                if c == 0:
                    print("primeira coluna")
                    probas = np.c_[np.zeros(n_instances), probas]
                elif c == max_y_train:
                    print("ultima coluna")
                    probas = np.c_[probas, np.zeros(n_instances)]
                else:
                    print("coluna meio")
                    probas = np.c_[probas[:,:c], np.zeros(n_instances), probas[:,c:]]
        
        return probas

    def select_data(self, X, y):

        # Check the X, y dimensions
        X, y = check_X_y(X, y, accept_sparse="csr")

        len_original_y = len(y)
        self.mask = np.ones(y.size, dtype=bool)

        self.classes_ = np.unique(y)
        
        alpha = self.fitting_alpha(X, y)
        beta = self.beta

        idx_choice_to_remove = self.select_end(alpha, beta)
        self._idx_redundant = idx_choice_to_remove
        self.mask[idx_choice_to_remove] = False

        noiseIdxToRemove = self.identifyNoiseByLowerNNEntropy(X, y)
        self._idx_noise = noiseIdxToRemove
        self.mask[noiseIdxToRemove] = False

        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])

        self.sample_indices_ = np.asarray(range(len(y)))[self.mask]
        self.reduction_ = 1.0 - float(len(self.y_))/len_original_y

        return self.X_, self.y_