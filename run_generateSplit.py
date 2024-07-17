from src.main.python.utils.general import get_data, get_splits, checkpoint_splits, translate_train_idxinfold
from src.main.python.utils.arguments import arguments
from src.main.python.utils.save_results import save_results
import argparse
from datetime import datetime
import time
import numpy as np
import gc
import io
import os
import pandas as pd
from collections import Counter
from src.main.python.iSel import cnn, enn, icf, lssm, lsbo, drop3, ldis, cdis, xldis, psdsp, ib3, cis, egdis, e2sc

import socket

import logging
import logging.config

logging.config.fileConfig('settings/logging.conf', defaults={'logfilename': f'resources/logs/{socket.gethostname()}.log'})
logger = logging.getLogger(__name__)

def get_selector(method: str):

    #Baselines
    if method == 'cnn':     return cnn.CNN()
    if method == 'enn':     return enn.ENN()
    if method == 'icf':     return icf.ICF()
    if method == 'lssm':    return lssm.LSSm()
    #if method == 'lsbo':    return lsbo.LSBo(args, fold)
    if method == 'lsbo':    return lsbo.LSBo()
    #if method == 'drop3':   return drop3.DROP3(args, fold, n_neighbors=3, loadenn=False)
    if method == 'drop3':   return drop3.DROP3()
    if method == 'ldis':    return ldis.LDIS()
    if method == 'cdis':    return cdis.CDIS()
    if method == 'xldis':   return xldis.XLDIS()
    if method == 'psdsp':   return psdsp.PSDSP()
    if method == 'ib3':     return ib3.IB3()
    if method == 'egdis':   return egdis.EGDIS()
    if method == 'cis':     return cis.CIS(task="atc")
    #proposed framework
    if method == 'e2sc':   return e2sc.E2SC()
    if method == 'e2sc-1':   return e2sc.E2SC(alphaMode="exact", betaMode='iterative')
    if method == 'e2sc-2':   return e2sc.E2SC(alphaMode="approximated", betaMode='heuristic')

    return None


def get_selection(X, y, fold, args):

    method = args.method

    total = Counter(y)
    logger.debug(f"Total instances number: {total}")

    selector = get_selector(method)

    selector.fit(X, y)

    logger.info("Result: ", Counter(y[selector.sample_indices_]))

    return selector.sample_indices_


def main():

    gc.collect()

    args, info = arguments()
    logger.info(str(args))

    print(f"{args.splitdir}/split_{args.folds}.pkl")
    splits_df = get_splits(f"{args.splitdir}/split_{args.folds}.pkl")

    splits_to_save = {c: [] for c in splits_df.columns if c.endswith("idxs")}
       
    #for f in range(args.folds):
    for f in range(1):

        logger.info("Fold {}".format(f))
        print("Fold {}".format(f))
        
        splits_to_save['test_idxs'].append(splits_df.loc[f].test_idxs)


        X_train, y_train, _, _, _ = get_data(args.inputdir, f)
        t = len(y_train)

        ti = time.time()

        idxs_docs = get_selection(X_train, y_train, f, args)

        s = len(y_train[idxs_docs])
        r = (t-s)/t

        info['time_for_reduce'].append(time.time() - ti)
        print(info['time_for_reduce'])
        info['original_len'].append(t)
        info['reduced_len'].append(s)
        info['reducion'].append(r)

        splits_to_save['train_idxs'].append(idxs_docs)

    logger.info(f"time: {np.mean(info['time_for_reduce'])}")
    logger.info(f"time std: {np.std(info['time_for_reduce'])}")
    logger.info(f"reducion: {np.mean(info['reducion'])}")
    logger.info(f"reducion std: {np.std(info['reducion'])}")

    splits_to_save_df = pd.DataFrame(data=splits_to_save)

    filename = f"{args.outputdir}/split_{args.folds}_{args.method}_idxinfold.pkl"

    checkpoint_splits(
        splits_df=splits_to_save_df,
        filename = filename
    )

    splits_to_save_df_traslated = translate_train_idxinfold(
        splits_to_save_df, splits_df)

    checkpoint_splits(
        splits_df=splits_to_save_df_traslated,
        filename=filename.replace("_idxinfold", "")
    )

    if args.save:
        save_results(args, info)
    
    print("END")
    exit()


if __name__ == '__main__':
    main()
