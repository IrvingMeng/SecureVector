#!/usr/bin/env python
import sys
import numpy as np
import argparse
import os
import time
import random
import hashlib
import resource


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
    P, r= np.load(file, allow_pickle=True)
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
    return int(r_prime==r)

  

folder = '/face/irving/eval_feats/template_protection/ironmask/lfw/'
pair_list ='/face/irving/data/ms1m_eval/lfw/pair.list'
feature_list = '/face/irving/eval_feats/magface_iresnet100/lfw_mf_10_110_0.45_0.8_20.list' 
alpha=16

with open(pair_list, 'r') as f:
    lines = f.readlines()

r_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   
features = load_features(feature_list)

n = len(lines) 
tmp = []
for i, line in enumerate(lines):
    file1, file2, _ = line.strip().split(' ')
    # load files
    feature1 = features[int(file1)]
    P, r = load_enrolled_file('{}/{}.npy'.format(folder, file2))
    tmp.append([feature1, P, r])
    if i % 1000 == 0:
        print('{}/{}'.format(i, n))    

print('[IronMask] Decrypting features...')
start = time.time()
for i, line in enumerate(lines):
    feature1, P, r = tmp[i]
    score = check_ironmask(feature1, P, r, alpha)
    if i % 1000 == 0:
        print('{}/{}'.format(i, n))        

duration = time.time() - start
r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r_init
print('total memory {}, total duration {}, encrypted {} features.\n'.format(r, duration, n))  