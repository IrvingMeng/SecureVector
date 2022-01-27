#!/usr/bin/env python
"""
Generate similarities directly from features and pair lists.
"""

import sys
import numpy as np
import argparse
import os
import time
import random


# parse the args
parser = argparse.ArgumentParser(description='Match in directly')
parser.add_argument('--feat_list', type=str)
parser.add_argument('--pair_list', default='', type=str, help='pair file')
parser.add_argument('--score_list', type=str,
                    help='a file which stores the scores')
args = parser.parse_args()


def load_features(feature_list):
    """
    load the features. 
    index (0,1,2,...), features
    """
    features = []
    with open(feature_list, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        feature = [float(e) for e in parts[1:]]
        feature = feature/np.linalg.norm(np.array(feature))
        features.append(feature)
    return features


def main(feat_list, pair_list, score_list):
    features = load_features(feat_list)
    with open(pair_list, 'r') as f:
        lines = f.readlines()
    n = len(lines)

    fw = open(score_list, 'w')
    for i, line in enumerate(lines):
        file1, file2, _ = line.strip().split(' ')
        score = np.dot(features[int(file1)], features[int(file2)])
        # measure time
        fw.write('{} {} {}\n'.format(file1, file2, score))
        if i % 1000 == 0:
            print('{}/{}'.format(i, n))
    fw.close()


if __name__ == '__main__':
    folder = '/'.join(args.score_list.split('/')[:-1])
    if not os.path.exists(folder):
        os.makedirs(folder)
    main(args.feat_list, args.pair_list, args.score_list)
