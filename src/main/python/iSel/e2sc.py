

import numpy as np
#import scipy.sparse as sp
import copy
import time

from sklearn.utils.validation import check_X_y
from sklearn.metrics import f1_score
from collections import Counter
from scipy import stats

from src.main.python.weakClassifiers.nmslibKNN import NMSlibKNNClassifier
from src.main.python.weakClassifiers.exactKNN import generateExactKNN
from src.main.python.iSel.e2sc_helpers import getCVSplit, print_stats, getCVSplitOnIteration
from src.main.python.iSel.base import InstanceSelectionMixin


class E2SC(InstanceSelectionMixin):
    """ Effective, Efficient, and Scalable Confidence-Based Instance Selection Framework (E2SC)
    Description:
    ==========

    We describe here the main contribution of [1]: the proposal of E2SC-IS - a novel two-step framework2 aimed at large datasets with a special focus on transformer-based architectures. E2SC-IS is a technique that satisfies the reduction-effectiveness-efficiency trade-off and is applicable in real-world scenarios, including datasets with thousands of instances.

    # Instantitation 1

    E2SC-IS ́s first step (fitting_alpha function) aims to assign a probability to each instance being removed from the training set (alpha). We adopt an KNN model to estimate the probability of removing instances, as it is considered a calibrated and computationally cheap model. Our first hypothesis (H1) is that high confidence (if the model is calibrated to the correct class, known in training) positively correlates with redundancy for the sake of building a classification model. Accordingly, we keep the hard-to-classify instances (probably located in the decision border regions), weighted by confidence, for the next step, in which we partially remove only the easy ones.
    
    As the second step of our method - beta estivative function - we propose to estimate a near-optimal reduction rate (beta parameter) that does not degrade the deep model's effectiveness by employing a validation set and a weak but fast classifier. Our second hypothesis (H2) is that we can estimate the effectiveness behavior of a robust model (deep learning) through the analysis and variation of selection rates in a weaker model. For this, again, we explore KNN. More specifically, we introduce an iterative method that statistically compares, using the validation set, the KNN model's effectiveness without any data reduction against the model with iterative data reduction rates. In this way, we can estimate a reduction rate that does not affect the KNN model's effectiveness. 
    
    Last, considering the output of these two steps together (end Mask function), beta% instances are randomly sampled, weighted by the alpha distribution, to be removed from the training set.

    # Instantiation 2

    To demonstrate the flexibility of our framework to cope with large datasets, we propose two modifications. The first one replaces the interactive strategy to optimize the parameter beta with a heuristic based on extracting statistical characteristics of the input dataset (heuristic beta). The second modification replaces the exact KNN with an approximate solution with logarithmic complexity, allowing a more scalable and efficient search for the nearest neighbors.

    Parameters:
    ===========

    alphaMode : {'exact', 'approximated'}, default='exact'
        Specifies the instantiation to be used in the first phase of the proposed framework.
        If 'exact' is given, the exact solution of the KNN model will be used. (scikit-learn)
        If 'approximated' is given, the approximate solution of the KNN with logarithmic complexity will be used. (NMSLIB)
    
    betaMode : {'iterative', 'prefixed'}, default='iterative'
        Specifies the instantiation to be used in the second phase of the proposed framework. 
        If 'iterative' is given, the beta reduction will be estimated by the iterative near-optimum procedure.
        If 'heuristic' is given, the beta reduction will be estimated by the heuristic considering the synthetic text characteristics and class distribution.
        If 'prefixed' is given, the beta reduction rate is supposed to be provided as a fixed reduction rate.
        

    beta : float, default=0.0
        Beta reduction rate. It is only significant when betaMode == 'prefixed'

    maxreduction : float, default=1.0
        Maximum reduction rate considered when betaMode == 'iterative'

    delta : float, default=0.05
        Incremental reduction rate step considered when betaMode == 'iterative'
        
    n_neighbors : int, default=10
        Number of neighbors considered by the KNN model.


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

    [1] Washington Cunha, Celso França, Guilherme Fonseca, Leonardo Rocha, and Marcos A. Gonçalves. An Effective, Efficient, and Scalable Confidence-based Instance Selection Framework for Transformer-based Text Classification. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR'23, New York, NY, USA, 2023. Association for Computing Machinery.

    [2] Washington Cunha, Felipe Viegas, Celso França, Thierson Rosa, Leonardo Rocha, and Marcos André Gonçalves. A Comparative Survey of Instance Selection Methods applied to NonNeural and Transformer-Based Text Classification. ACM Computing Surveys, 2023.

    
    Example
    =======

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from src.main.python.iSel import e2sc
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> selector = e2sc.E2SC()
    >>> idx = selector.sample_indices_
    >>> X_train_selected, y_train_selected =  X_train[idx], y_train[idx]
    >>> print('Resampled dataset shape %s' % Counter(y_train_selected))
    Resampled dataset shape Counter({1: 36, 0: 14})
    """

    def __init__(self, alphaMode="exact", 
                       betaMode="iterative",
                       beta = 0.0,
                       maxreduction = 1.0,
                       delta = 0.05,
                       n_neighbors = 0):
        
        if alphaMode not in ["exact", "approximated"]:
            raise ValueError("Invalid alphaMode type. Expected one of: %s" % ["exact", "approximated"])
        
        if betaMode not in ["iterative", "heuristic" ,"prefixed"]:
            raise ValueError("Invalid betaMode type. Expected one of: %s" % ["iterative", "heuristic", "prefixed"])
        
        if betaMode == "prefixed" and beta == 0.0:
            raise ValueError("beta should be greater than 0.0 when betaMode is setted to prefixed")
        
        self.alphaMode = alphaMode
        self.betaMode = betaMode
        self.beta = beta
        self.maxreduction = maxreduction
        self.delta = delta * 100
        
        self.sample_indices_ = []

        if beta > 0.0:
            self.betaMode = "prefixed"

        if self.alphaMode == "exact":
            self.fitting_alpha = self.exact_fitting_alpha
        elif self.alphaMode == "approximated":
            self.fitting_alpha = self.approximated_fitting_alpha

        self.n_neighbors = n_neighbors




    def exact_fitting_alpha(self, X, y):

        # Generation  Train - Val validation for further analysis in the iterative beta process
        cvSplit = getCVSplit(X, y)

        # Generates the results considering the entire training set
        y_pred_full_training, macro_val_list_full_training, max_proba_by_instance_full_training, self.n_neighbors = generateExactKNN(
            X, y, cvSplit, doGridSearch=True, n_neighbors=self.n_neighbors)
        
        macro_mean_full_training, macro_ic_full_training = print_stats(
            macro_val_list_full_training)
        
        # Setting the removal probability of wrong predicted instances as zero
        corret_predicted = 0
        correctPredictedProba = np.zeros(y.size, dtype=float)
        for i in range(y.size):
            if y[i] == y_pred_full_training[i]:
                correctPredictedProba[i] = max_proba_by_instance_full_training[i]
                corret_predicted += 1

        # Normalazing the results to reach the the alpha distribution
        alpha = correctPredictedProba / np.sum(correctPredictedProba)
        correctPredictedRate = corret_predicted/y.size

        # Saving some info for further analysis in the iterative beta process
        self.exactInfo = {
            "macro_mean_full_training":macro_mean_full_training,
            "macro_ic_full_training":macro_ic_full_training,
            "macro_val_list_full_training":macro_val_list_full_training,
            "correctPredictedRate":correctPredictedRate,
            "cvSplit":cvSplit
        }

        return alpha


    def approximated_fitting_alpha(self, X, y):

        # Setting the approximated KNN solution
        classifier = NMSlibKNNClassifier(n_neighbors=10, n_jobs=10)
        classifier.fit(X, y)

        # Predicting the probabilities using the approximated KNN solution
        pred, proba = classifier.predict_y_and_maxproba_for_X_train(X)

        if f1_score(y, pred,average='micro') < self.beta:
            raise ValueError("ERROR. KNN accuracy < beta")

        # Setting the removal probability of wrong predicted instances as zero
        correctPredictedProba = copy.copy(proba)
        correctPredictedProba[pred != y] = 0.
        # Normalazing the results to reach the the alpha distribution
        correctPredictedProba = correctPredictedProba / np.sum(correctPredictedProba)

        #alpha = correctPredictedProba
        return correctPredictedProba

    def select_end(self, alpha, beta):

        # Choosing the instances to be removed based on either alpha distibution and beta rate.
        n_training_samples = len(alpha)
        n_samples_to_remove = int(n_training_samples * beta)
        
        idx_choice_to_remove = np.random.choice(a=list(range(n_training_samples)),
                                                size=n_samples_to_remove,
                                                replace=False,
                                                p=alpha)
        
        return idx_choice_to_remove



    def iterative_beta(self, X, y, alpha):

        # Setting the percents to be analized
        percents = [round(x, 2) for x in (np.arange(self.delta, 100, self.delta) / 100)]

        percents = [p for p in percents if (
            (p + 0.01) < self.exactInfo["correctPredictedRate"])] 
        
        percents = [p for p in percents if p <= self.maxreduction]
        #print(percents)

        t = time.time()
        times = []
        equivalentPercents = []
        intercept = []
        flagPrevEquivalent = True
        finalResult = [0.90, 0]

        for percent_to_remove in percents:

            #print(f"Trying to remove {percent_to_remove:.2f} . Sel = {1.0-percent_to_remove:.2f}")
            try:
                cvSplit_onIteration = getCVSplitOnIteration(alpha, percent_to_remove, self.exactInfo["cvSplit"])
            except:
                break

            # generates the result considering the exact knn solution
            _, macro_val_list_onIteration, _, _ = generateExactKNN(X, y, cvSplit_onIteration, doGridSearch=False, n_neighbors=self.n_neighbors)
            
            times.append(f"{time.time()-t:.2f}")

            macro_mean_onIteration, macro_ic_onIteration = print_stats(
                macro_val_list_onIteration)
            
            # save the percent if the result is intercepts than the model trained considering the entire training set (instersection) 
            if macro_mean_onIteration + macro_ic_onIteration > self.exactInfo["macro_mean_full_training"] - self.exactInfo["macro_ic_full_training"]:
                intercept.append(1.0-percent_to_remove)

            # save the percent if the result is greater than the model trained considering the entire training set (statistically equivalent) 
            _, p_macro = stats.ttest_ind(
                self.exactInfo["macro_val_list_full_training"], macro_val_list_onIteration)
            if p_macro > 0.05 or macro_mean_onIteration > self.exactInfo["macro_mean_full_training"]:
                equivalentPercents.append(1.0-percent_to_remove)
                if flagPrevEquivalent:
                    finalResult = [1.0-percent_to_remove, times[-1]]
            else:
                break
                #flagPrevEquivalent = False


        if finalResult[0] > self.maxreduction:
            finalResult = [1.0-self.maxreduction, times[-1]]


        return 1.0-finalResult[0]
    
    def heuristic_beta(self, X, y):

        classDist = Counter(y)
        minor, major = min(classDist.values()), max(classDist.values())
        
        if minor/(minor+major) < 0.25: # imbalanced or extremely imbalanced
            return 0.25

        if np.mean(X.getnnz(axis=1)) < 100: # average density is low (less than 100)
            return 0.25

        return 0.50


    def estimating_beta(self, X, y, alpha):

        if self.betaMode == 'prefixed':
            return self.beta
        elif self.betaMode == 'iterative':
            return self.iterative_beta(X, y, alpha)
        elif self.betaMode == 'heuristic':
            return self.heuristic_beta(X, y)

    def select_data(self, X, y):

        # Check the X, y dimensions
        X, y = check_X_y(X, y, accept_sparse="csr")

        len_original_y = len(y)
        self.mask = np.ones(y.size, dtype=bool)

        self.classes_ = np.unique(y)
        
        alpha = self.fitting_alpha(X, y)

        beta = self.estimating_beta(X, y, alpha)

        idx_choice_to_remove = self.select_end(alpha, beta)
        self.mask[idx_choice_to_remove] = False

        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])

        self.sample_indices_ = np.asarray(range(len(y)))[self.mask]
        self.reduction_ = 1.0 - float(len(self.y_))/len_original_y

        return self.X_, self.y_