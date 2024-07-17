
from sklearn.model_selection import StratifiedKFold
from scipy.stats import t as qt
import numpy as np

def getCVSplit(X, y):
    skf = StratifiedKFold(n_splits=5)

    cvSplit = []
    for train_index, test_index in skf.split(X, y):
        cvSplit.append((train_index, test_index))

    return cvSplit

def getCVSplitOnIteration(correctPredictedProba, percent_to_remove, cvSplit):

    cvSplit_onIteration = []
    for train_index, test_index in cvSplit:

        n_training_samples = train_index.size

        n_samples_to_remove = int(
            n_training_samples * percent_to_remove)

        proba = correctPredictedProba[train_index] / \
            sum(correctPredictedProba[train_index])

        idx_choice_to_remove = np.random.choice(a=list(range(n_training_samples)),
                                                size=n_samples_to_remove,
                                                replace=False,
                                                p=proba)

        idx_choice_to_remove = list(sorted(idx_choice_to_remove))

        mask = np.ones(n_training_samples, dtype=bool)
        mask[idx_choice_to_remove] = False

        cvSplit_onIteration.append(
            (train_index[mask], test_index))

    return cvSplit_onIteration

def logger(message):
    #with open("/home/waashk/atcisel/resources/logs/KNN_v7_Test.log", "a") as arq:
    #    arq.write(message+"\n")
    print(message)

def print_stats(mlist):
        #print(micro_list)
        folds = len(mlist)
        med = np.mean(mlist)*100
        error = abs(qt.isf(0.975, df=(folds-1))) * \
            np.std(mlist, ddof=1)/np.sqrt(len(mlist))*100

        return med, error