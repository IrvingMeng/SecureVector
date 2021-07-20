#!/usr/bin/env python

import numpy as np
import phe.paillier as paillier
from gmpy2 import mpz
import argparse
import os
import time
import random
import resource
from itertools import repeat


def load_enrolled_file(file):
    c_f, C_tilde_f= np.load(file, allow_pickle=True)
    return c_f, C_tilde_f

def decrypt_sum(C_tilde_x, C_tilde_y):   
    C_z = private_key.decrypt(C_tilde_x + C_tilde_y)
    return C_z

def decode_uvw(C_f, K, L):
    u_list, v_list = [], []
    for i in range(K):
        next_C_f = C_f//(4*L)
        u_list.append(C_f - (4*L)*next_C_f)
        C_f = next_C_f
    for i in range(K):
        next_C_f = C_f//(4*L)
        v_list.append(C_f - (4*L)*next_C_f)
        C_f = next_C_f
    w_f = C_f
    u_list.reverse()
    v_list.reverse()
    return u_list, v_list, int(w_f)


def calculate_sim(c_x, c_y, C_tilde_x, C_tilde_y, K, L, M):
    # decrypt 
    C_z = decrypt_sum(C_tilde_x, C_tilde_y)

    # generate bar_c_xy
    c_xy = c_x*c_y
    n = len(c_x)    
    bar_c_xy = [sum(c_xy[i:i+n//K]) for i in range(0, n, n//K)]
    
    # recover u_list, v_list, w
    u_list, v_list, w_z = decode_uvw(C_z, K, L)
    s_list = [1 if v%2==0 else -1 for v in v_list]    
    # calculate the score
    W_z = np.e**((w_z - 2**15 * L**8)/(2**14 * L**7*M))
    score = W_z * sum([bar_c_xy[i]/(s_list[i] * np.e**((u_list[i]-2*L)/M)) for i in range(K)])
    
    return score


folder = '/face/irving/eval_feats/template_protection/invisibleface/lfw/'
pair_list ='/face/irving/data/ms1m_eval/lfw/pair.list'

with open(pair_list, 'r') as f:
    lines = f.readlines()

K=16
key_size=1024
L = int(np.ceil(2**(key_size/(2*K+9)-2) - 1))
M = L/128
security_level = 2*K + K*np.log2(L)

print('K: {}   L: {}   M: {}'.format(K, L, M))
print('the security level is: {}'.format(security_level))

r_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   
private_key = np.load('/face/irving/eval_feats/template_protection/invisibleface/privatekey_{}.npy'.format(key_size), allow_pickle=True)[0]

n = len(lines) 
tmp = []
for i, line in enumerate(lines):
    file1, file2, _ = line.strip().split(' ')
    # load files
    c_x, C_tilde_x = load_enrolled_file('{}/{}.npy'.format(folder, file1))
    c_y, C_tilde_y = load_enrolled_file('{}/{}.npy'.format(folder, file2))
    tmp.append([c_x, C_tilde_x, c_y, C_tilde_y])
    if i % 1000 == 0:
        print('{}/{}'.format(i, n))    


start = time.time()


for i, line in enumerate(lines):
    c_x, C_tilde_x, c_y, C_tilde_y = tmp[i]
    score = calculate_sim(c_x, c_y, C_tilde_x, C_tilde_y, K, L, M)
    if i % 1000 == 0:
        print('{}/{}'.format(i, n))        

duration = time.time() - start
r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r_init
print('total memory {}, total duration {}, encrypted {} features.\n'.format(r, duration, n))  


