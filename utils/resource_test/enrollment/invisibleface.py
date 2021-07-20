#!/usr/bin/env python
import sys
import numpy as np
import phe.paillier as paillier
from gmpy2 import mpz
import argparse
import os
import time
import random
from itertools import repeat
import shutil
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

def enroll(feature, K, L, M, public_key):
    """
    enroll a feature
    """    
    u_list = [int(e) for e in np.random.rand(K)*(2*L)]
    v_list = [int(e) for e in np.random.rand(K)*(2*L)]
    s_list = [1 if v%2==0 else -1 for v in v_list]

    # generate c_f    
    n = len(feature)
    scale = [s_list[i] * np.e**((u_list[i]-L)/M)  for i in range(K)]
    b_f = [x for item in  scale for x in repeat(item, n//K)] * feature
    W_f = np.linalg.norm(b_f)
    c_f = b_f/W_f

    # encrypt
    base = [(4*L)**(K-1-i) for i in range(K)]
    w_f = int((np.log(W_f) + L/M)/(2*L/M) * 2**15 * L**8)
    C_f = np.dot(u_list, base) + \
            np.dot(v_list, base) * (4*L)**(K) + \
              w_f * (4*L)**(2*K)
    
    C_tilde_f = public_key.encrypt(C_f)
    return [c_f, C_tilde_f]


K=16
key_size=256
public_key='/face/irving/eval_feats/template_protection/invisibleface/publickey'
feature_list = '/face/irving/eval_feats/magface_iresnet100/lfw_mf_10_110_0.45_0.8_20.list'


L = int(np.ceil(2**(key_size/(2*K+9)-2) - 1))
M = L/128
security_level = 2*K + K*np.log2(L)

print('K: {}   L: {}   M: {}'.format(K, L, M))
print('the security level is: {}'.format(security_level))

features = load_features(feature_list)
n, dim = len(features), len(features[0])

r_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
start = time.time()
print('[InvisibleFace] Encrypting features...') 

publickey = np.load('{}_{}.npy'.format(public_key, key_size), allow_pickle=True)[0]  

results = []
for i, feature in enumerate(features):        
    result = enroll(feature, K, L, M, publickey)
    results.append(result)    
    if i % 1000 == 0:
        print('{}/{}'.format(i, n))     

duration = time.time() - start
r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r_init
print('total memory {}, total duration {}, encrypted {} features.\n'.format(r, duration, n))






