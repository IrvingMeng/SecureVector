#!/usr/bin/env python
"""
This is the "root" code, only it can get access to the private key.
    In: paths to two enrolled features
    Out: Similarity scores
useage:
    generate keys:
        python crypo_system.py --genkey 1 --key_size 1024 
    calculate similarities:
        python crypo_system.py --key_size 1024 --K 128 --folder $F --pair_list $P --score_list $S
"""
import sys
import math
import numpy as np
import phe.paillier as paillier
from gmpy2 import mpz
import argparse
import os
import time
import random
import resource
from itertools import repeat
from joblib import Parallel, delayed

# parse the args
parser = argparse.ArgumentParser(description='Match in SecureVector')
parser.add_argument('--folder', default='', type=str, help='fold which stores the encrypted features')
parser.add_argument('--pair_list', default='', type=str, help='pair file')
parser.add_argument('--score_list', type=str, help='a file which stores the scores')
parser.add_argument('--K', default=128, type=int)
parser.add_argument('--key_size', default=1024, type=int)
parser.add_argument('--genkey', default=0, type=int)
args = parser.parse_args()    

if args.genkey == 1:
    pubkey, prikey = paillier.generate_paillier_keypair(n_length=args.key_size)
    np.save('libs/SecureVector/keys/privatekey_{}.npy'.format(args.key_size), [prikey])
    np.save('libs/SecureVector/keys/publickey_{}.npy'.format(args.key_size), [pubkey])
    exit(1)
else:
    private_key = np.load('libs/SecureVector/keys/privatekey_{}.npy'.format(args.key_size), allow_pickle=True)[0]

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
    start = time.time()
    C_z = decrypt_sum(C_tilde_x, C_tilde_y)
    duration_cypher = time.time() - start

    # generate bar_c_xy
    start = time.time()
    c_xy = c_x*c_y
    n = len(c_x)    
    bar_c_xy = [sum(c_xy[i:i+n//K]) for i in range(0, n, n//K)]
    
    # recover u_list, v_list, w
    u_list, v_list, w_z = decode_uvw(C_z, K, L)
    s_list = [1 if v%2==0 else -1 for v in v_list]    
    # calculate the score
    W_z = np.e**((w_z - 2**15 * L**8)/(2**14 * L**7*M))
    score = W_z * sum([bar_c_xy[i]/(s_list[i] * np.e**((u_list[i]-2*L)/M)) for i in range(K)])
    duration_plain = time.time() - start

    return score, [duration_plain, duration_cypher]

def chunkify(fname,size=1024*1024):
    fileEnd = os.path.getsize(fname)
    with open(fname,'rb') as f:
        chunkEnd = f.tell()
        while True:
            chunkStart = chunkEnd
            f.seek(size,1)
            f.readline()
            chunkEnd = f.tell()
            yield chunkStart, chunkEnd - chunkStart
            if chunkEnd > fileEnd:
                break

def process_lines(chunk_info_list, pair_list, folder, K, L, M, i):
    score_list = []
    durations_list = []
    partid_lineinfo_map = {}
    with open(pair_list,'r') as f:
        for j in range(len(chunk_info_list)):
            chunkStart, chunkSize = chunk_info_list[j]
            f.seek(chunkStart)
            lines = f.read(chunkSize).splitlines()
            for line in lines:
                file1, file2, _ = line.strip().split(' ')
                # load files
                c_x, C_tilde_x = load_enrolled_file('{}/{}.npy'.format(folder, file1))
                c_y, C_tilde_y = load_enrolled_file('{}/{}.npy'.format(folder, file2))
                # here you need to check if c_x, c_y, C_tilde_x, C_tilde_y are similar
                # if they are similar, the code should refuse to encrypt the results
                score, durations = calculate_sim(c_x, c_y, C_tilde_x, C_tilde_y, K, L, M)
                score_list.append((file1, file2, score))
                durations_list.append((file1, file2, durations))

    partid_lineinfo_map[i] = [score_list, durations_list]

    return partid_lineinfo_map

def main(folder, pair_list, score_list,  K, L, M):
    # load pair_file
    with open(pair_list, 'r') as f:
        lines = f.readlines()

    fw = open(score_list, 'w')

    print('[SecureVector] Decrypting features...')    
    start = time.time()
    duration_plain = []
    duration_cypher = []    

    n = len(lines)
    if n < 100000:
        for i, line in enumerate(lines):
            file1, file2, _ = line.strip().split(' ')
            # load files
            c_x, C_tilde_x = load_enrolled_file('{}/{}.npy'.format(folder, file1))
            c_y, C_tilde_y = load_enrolled_file('{}/{}.npy'.format(folder, file2))
            # here you need to check if c_x, c_y, C_tilde_x, C_tilde_y are similar
            # if they are similar, the code should refuse to encrypt the results
            score, durations = calculate_sim(c_x, c_y, C_tilde_x, C_tilde_y, K, L, M)
            # measure time
            duration_plain.append(durations[0])
            duration_cypher.append(durations[1])      
            fw.write('{} {} {}\n'.format(file1, file2, score))
            if i % 1000 == 0:
                print('{}/{}'.format(i, n))
    else:
        # Paralel Generate the scores.
        num_jobs = 12
        chunk_info_list =list(chunkify(pair_list))
        lnum = len(chunk_info_list)
        idxs = list(range(0,lnum, math.ceil(lnum/num_jobs)))
        idxs.append(lnum)
        num_jobs = len(idxs) - 1
        result_list = Parallel(n_jobs=num_jobs, verbose=100)(delayed(process_lines)(
            chunk_info_list[idxs[i]:idxs[i+1]], pair_list, folder, K, L, M, i) for i in range(num_jobs))
        # concat in order
        all_partid_lineinfo_map = {}
        for (partid_lineinfo_map)  in result_list:
            for partid, info in partid_lineinfo_map.items():
                all_partid_lineinfo_map[partid] = info

        i = 0
        for j in range(num_jobs):
            score_list, durations_list = all_partid_lineinfo_map[j]
            assert len(score_list) == len(durations_list)
            for lineid, scoreinfo in enumerate(score_list):
                file1, file2, score = scoreinfo
                fw.write('{} {} {}\n'.format(file1, file2, score))
                _, _, durations = durations_list[lineid]
                duration_plain.append(durations[0])
                duration_cypher.append(durations[1]) 
                if i % 1000 == 0:
                    print('{}/{}'.format(i, n))
                i += 1

    fw.close()
    
    duration = time.time() - start
    print('otal duration {}, permutation duration {}, paillier duration {}, calculate {} pairs.\n'.format(duration, sum(duration_plain), sum(duration_cypher), n))    


if __name__ == '__main__':        
    L = int(np.ceil(2**(args.key_size/(2*args.K+9)-2) - 1))
    M = L/128
    security_level = 2*args.K + args.K*np.log2(L)

    assert L > 1
    # print('K: {}   L: {}   M: {}'.format(args.K, L, M))
    # print('the security level is: {}'.format(security_level))

    main(args.folder, args.pair_list, args.score_list, args.K, L, M)
