import sys
import numpy as np
import phe.paillier as paillier
from gmpy2 import mpz
import os
import time
import random
from itertools import repeat


def load_enrolled_file(file):
    c_f, C_tilde_f= np.load(file, allow_pickle=True)
    return c_f, C_tilde_f

def decrypt_sum(C_tilde_x, C_tilde_y, private_key):   
    start = time.time() 
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


def enroll(feature, K, L, M, public_key):
    """
    enroll a feature
    """
    start = time.time()
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
    duration1 = time.time() - start

    start = time.time()
    C_tilde_f = public_key.encrypt(C_f)
    duration2 = time.time() - start 
    return [c_f, C_tilde_f], [duration1, duration2]