#!/usr/bin/env python
import sys
import numpy as np
import argparse
import os
import time
import random
import hashlib
import resource

# parse the args
parser = argparse.ArgumentParser(description='Match in IronMask')
parser.add_argument('--folder', default='', type=str,
                    help='fold which stores the encrypted features')
parser.add_argument('--pair_list', default='', type=str, help='pair file')
parser.add_argument('--score_list', type=str,
                    help='a file which stores the scores')
parser.add_argument('--alpha', type=int, default=16)
parser.add_argument('--feat_list', type=str)
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


def load_enrolled_file(file):
    P, r = np.load(file, allow_pickle=True)
    return P, r


def decode(feature, alpha=16):
    """
    map a feature to the corresponding code
    """
    base_ele = 1/np.sqrt(alpha)
    abs_feature = np.abs(feature)
    indexes = np.argsort(abs_feature)[::-1]
    code = np.zeros(len(feature))
    for i in range(alpha):
        code[indexes[i]] = base_ele
        if feature[indexes[i]] < 0:
            code[indexes[i]] = code[indexes[i]] * (-1)
    return code


def check_ironmask(feature, P, r, alpha):
    """
    return 1 if feature and P,r is from same id
    """
    start = time.time()
    c_prime = decode(np.dot(P, feature), alpha)

    hash_func = hashlib.md5()
    hash_func.update(c_prime.tobytes())
    r_prime = hash_func.hexdigest()
    return int(r_prime == r), time.time() - start


def main(folder, feat_list, pair_list, score_list, alpha):
    # load pair_file
    features = load_features(feat_list)
    with open(pair_list, 'r') as f:
        lines = f.readlines()

    fw = open(score_list, 'w')

    print('[IronMask] Decrypting features...')
    r_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.time()
    duration_plain = []
    n = len(lines)
    for i, line in enumerate(lines):
        file1, file2, _ = line.strip().split(' ')
        # load files
        feature1 = features[int(file1)]
        P, r = load_enrolled_file('{}/{}.npy'.format(folder, file2))

        score, duration = check_ironmask(feature1, P, r, alpha)
        duration_plain.append(duration)
        fw.write('{} {} {}\n'.format(file1, file2, score))
        if i % 1000 == 0:
            print('{}/{}'.format(i, n))
    fw.close()

    duration = time.time() - start
    print('total duration {}, ironmask duration {}, calculate {} pairs.\n'.format(
        duration, sum(duration_plain), n))


if __name__ == '__main__':
    main(args.folder, args.feat_list, args.pair_list, args.score_list, args.alpha)
