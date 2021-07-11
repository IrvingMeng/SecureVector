#!/usr/bin/env python
"""
This is the "root" code, only it can get access to the private key.
    In: paths to two enrolled features
    Out: Similarity scores
useage:
    generate keys:
        python crypo_system.py --genkey 1 --key_size 1024 
    calculate similarities:
        python crypo_system.py --key_size 1024 --file1 $F1 --file2 $F2 --K 128
"""
import sys
import numpy as np
import phe.paillier as paillier
from gmpy2 import mpz
import argparse
import os
import time
import random
from itertools import repeat

# parse the args
parser = argparse.ArgumentParser(description='Match in InvisibleFace')
parser.add_argument('--file1', default='', type=str)
parser.add_argument('--file2', default='', type=str)
parser.add_argument('--K', default=128, type=int)
parser.add_argument('--key_size', default=1024, type=int)
parser.add_argument('--genkey', default=0, type=int)
args = parser.parse_args()    

if args.genkey == 1:
    pubkey, prikey = paillier.generate_paillier_keypair(n_length=args.key_size)
    np.save('/face/irving/eval_feats/invisibleface/privatekey_{}.npy'.format(args.key_size), [prikey])
    np.save('/face/irving/eval_feats/invisibleface/publickey_{}.npy'.format(args.key_size), [pubkey])
    exit(1)
else:
    private_key = np.load('/face/irving/eval_feats/invisibleface/privatekey_{}.npy'.format(args.key_size), allow_pickle=True)[0]

def load_enrolled_file(file):
    c_f, C_tilde_f= np.load(file, allow_pickle=True)
    return c_f, C_tilde_f

def decrypt_sum(C_tilde_x, C_tilde_y):   
    start = time.time() 
    C_z = private_key.decrypt(C_tilde_x + C_tilde_y)
    print('decrypt {}'.format(time.time()-start))
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


def main(file1, file2, K, L, M):
    # load files
    c_x, C_tilde_x = load_enrolled_file(file1)
    c_y, C_tilde_y = load_enrolled_file(file2)
    # here you need to check if c_x, c_y, C_tilde_x, C_tilde_y are similar
    # if they are similar, the code should refuse to encrypt the results

    start = time.time()
    # generate bar_c_xy
    c_xy = c_x*c_y
    n = len(c_x)    
    bar_c_xy = [sum(c_xy[i:i+n//K]) for i in range(0, n, n//K)]
    
    # decrypt 
    C_z = decrypt_sum(C_tilde_x, C_tilde_y)
    
    # recover u_list, v_list, w
    u_list, v_list, w_z = decode_uvw(C_z, K, L)
    s_list = [1 if v%2==0 else -1 for v in v_list]    

    # calculate the score
    W_z = np.e**((w_z - 2**15 * L**8)/(2**14 * L**7*M))
    score = W_z * sum([bar_c_xy[i]/(s_list[i] * np.e**((u_list[i]-2*L)/M)) for i in range(K)])

    print(time.time() - start)    
    print(score)    
    return score


# load the file
if __name__ == '__main__':        
    L = int(np.ceil(2**(args.key_size/(2*args.K+9)-2) - 1))
    M = L/128
    security_level = 2*args.K + args.K*np.log2(L)

    assert L > 1
    print('K: {}   L: {}   M: {}'.format(args.K, L, M))
    print('the security level is: {}'.format(security_level))

    main(args.file1, args.file2, args.K, L, M)
