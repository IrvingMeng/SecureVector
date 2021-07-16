#!/usr/bin/env python
"""
1v1 evaluation. 
    1. use 10-fold cross validation.
    2. [NOT] use mean substraction.
"""
import math
import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--pair_list', type=str,
                    help='opensource pair list.')
parser.add_argument('--score_list', type=str,
                    help='opensource score list.')                    

def distance_(embeddings0, embeddings1):
    # Distance based on cosine similarity
    dot = np.sum(np.multiply(embeddings0, embeddings1), axis=1)
    norm = np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1)
    # shaving
    similarity = dot/norm
    similarity = np.clip(similarity, -1., 1.)
    dist = np.arccos(similarity) / math.pi
    return dist

def calculate_roc(thresholds, dist, actual_issame, nrof_folds=1):
    nrof_pairs = len(actual_issame)
    nrof_thresholds = len(thresholds)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    # Find the best threshold for the fold
    acc_train = np.zeros((nrof_thresholds))
    for threshold_idx, threshold in enumerate(thresholds):
        _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist, actual_issame)
    best_threshold_index = np.argmax(acc_train)
    fold_idx = 0
    for threshold_idx, threshold in enumerate(thresholds):
        tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist, actual_issame)
    _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist, actual_issame)

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

class LFold:
    def __init__(self, n_splits = 2, shuffle = False):
        self.n_splits = n_splits
        if self.n_splits>1:
            self.k_fold = KFold(n_splits = n_splits, shuffle = shuffle)

    def split(self, indices):
        if self.n_splits>1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]

def perform_1v1_eval(targets, dists):
    targets = np.vstack(targets).reshape(-1,)
    dists = np.vstack(dists).reshape(-1,)

    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy = calculate_roc(thresholds, dists, targets)
    print('    Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    resultline='%2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy))

def load_pair_score(pair_list, score_list):
    with open(pair_list, 'r') as f:
        pair_lines = f.readlines()
    with open(score_list, 'r') as f:
        score_lines = f.readlines()
    assert len(pair_lines) == len(score_lines)

    # load pair score
    targets, dists = [], []
    for i in range(len(pair_lines)):
        parts1 = pair_lines[i].strip().split(' ')
        parts2 = score_lines[i].strip().split(' ')
        assert parts1[0] == parts2[0]
        assert parts1[1] == parts2[1]
        is_same = int(parts1[2])
        dist = np.arccos(float(parts2[2])) / math.pi        
        # collect
        targets.append(is_same)
        dists.append(dist)
    return targets, dists


def eval(pair_list, score_list):
    targets, dists = load_pair_score(pair_list, score_list)
    perform_1v1_eval(targets, dists)


def main():
    args = parser.parse_args()
    eval(args.pair_list, args.score_list)

if __name__ == '__main__':
    main()
