#!/usr/bin/env python
import sys
import numpy as np
import argparse
import os
import time
import shutil
import math
from scipy.stats import ortho_group
from numpy.random import default_rng
import hashlib
import resource

rng = default_rng()

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


def gen_code(alpha=16, n=512):
    """
    generate a code with dim n and alpha non zeros entries
    """
    base_ele = 1/np.sqrt(alpha)
    indexes = rng.choice(n, size=alpha, replace=False)
    symbols = rng.choice([-1, 1], size=alpha, replace=True)
    code = np.zeros(n)
    for i, index in enumerate(indexes):
        code[index] = base_ele * symbols[i]
    return code


def compute_rotation(t, c):
    """
    compute a rotation matrix R which has Rt = c
    """
    # check if normalized and dim is the same
    assert_error = 1e-5
    assert(abs(np.linalg.norm(t)-1) < assert_error)
    assert(abs(np.linalg.norm(c)-1) < assert_error)
    assert(len(t) == len(c))
    
    # here starts
    I = np.identity(len(t))
    w = c - np.dot(t, c)*t
    w = w / np.linalg.norm(w)
    
    cos_theta = np.dot(t, c)
    sin_theta = math.sin(math.acos(cos_theta))
    
    R = I - np.outer(t, t) - np.outer(w, w) + \
        (np.outer(t, t) + np.outer(w, w)) * cos_theta + \
        (np.outer(w, t) - np.outer(t, w)) * sin_theta
    return R    


def enroll_ironmask(feature, alpha):    
    dim = len(feature)
    c = gen_code(alpha=alpha, n=dim)

    feature = feature/np.linalg.norm(feature)

    Q = ortho_group.rvs(dim)
    R = compute_rotation(np.dot(Q,feature), c)
    P = np.dot(R, Q)
    # hash
    hash_func = hashlib.md5()
    hash_func.update(c.tobytes())
    r = hash_func.hexdigest()
    return [P, r]


alpha=16
feature_list = '/face/irving/eval_feats/magface_iresnet100/lfw_mf_10_110_0.45_0.8_20.list'

# print('loading features...')
features = load_features(feature_list)
n, dim = len(features), len(features[0])
# L_list = [i for i in range(0, 2*L)]    

print('[IronMask] Encrypting features...')  
r_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  
start = time.time()
results=[]

for i, feature in enumerate(features):
    result = enroll_ironmask(feature, alpha)        
    results.append(result)
    if i % 1000 == 0:
        print('{}/{}'.format(i, n))    
duration = time.time() - start
r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r_init
print('total memory {}, total duration {}, encrypted {} features.\n'.format(r, duration, n))


