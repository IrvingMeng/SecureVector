#!/usr/bin/env python
import argparse
import os
import cv2
import math
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--pair_list', type=str,
                    help='opensource pair list.')
parser.add_argument('--score_list', type=str,
                    help='opensource score list.')                    

def perform_1vn_eval(label, scores):
    fpr, tpr, _ = roc_curve(label, scores)
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)
    x_labels = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
    to_print = ''
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))

        print('  {:0.4f}'.format(tpr[min_index]))

        to_print = to_print + '  {:0.4f}'.format(tpr[min_index])
    
    print(to_print)

def load_pair_score(pair_list, score_list):
    with open(pair_list, 'r') as f:
        pair_lines = f.readlines()
    with open(score_list, 'r') as f:
        score_lines = f.readlines()
    assert len(pair_lines) == len(score_lines)

    targets, scores = [], []
    for i in range(len(pair_lines)):
        parts1 = pair_lines[i].strip().split(' ')
        parts2 = score_lines[i].strip().split(' ')
        assert parts1[0] == parts2[0]
        assert parts1[1] == parts2[1]
        is_same = int(parts1[2])
        score =  float(parts2[2])
        targets.append(is_same)
        scores.append(score)
    return targets, scores


def eval(pair_list, score_list):
    labels, scores = load_pair_score(pair_list, score_list)
    perform_1vn_eval(labels, scores)


def main():
    args = parser.parse_args()
    eval(args.pair_list, args.score_list)

if __name__ == '__main__':
    main()