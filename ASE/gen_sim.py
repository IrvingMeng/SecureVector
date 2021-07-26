#!/usr/bin/env python
import sys
import math
import numpy as np
import argparse
import os
import time
import random
import resource
from joblib import Parallel, delayed

# parse the args
parser = argparse.ArgumentParser(description='Match in ASE')
parser.add_argument('--folder', default='', type=str, help='fold which stores the encrypted features')
parser.add_argument('--pair_list', default='', type=str, help='pair file')
parser.add_argument('--score_list', type=str, help='a file which stores the scores')
args = parser.parse_args()    

def load_enrolled_file(file):
    d, basis_d= np.load(file, allow_pickle=True)
    return d, basis_d

def convert_orth_basis(basis):
    """
    generate orthogonal basis 
    """
    u_list = []
    for base in basis:
        proj_base = base
        for u in u_list:
            proj_base = proj_base - np.dot(proj_base, u)*u
        proj_base = proj_base / np.linalg.norm(proj_base)
        u_list.append(proj_base)
    return u_list


def ortho_proj(e, d, basis):
    """
    calculate orthogonal projection of e onto d + span(basis)
    """
    u_list = convert_orth_basis(basis)
    relative_ = e - d
    proj_e = d
    for u in u_list:
        proj_e = proj_e +  np.dot(relative_, u) * u    
    return proj_e


def dist_p_to_s(e, d, basis_d):
    """
    point-to-subspace distance
    """    
    proj_e = ortho_proj(e, d, basis_d)
    dist = np.linalg.norm(e - proj_e)
    return dist


def dist_s_to_s(d, basis_d, e, basis_e):
    """
    subspace-to-subspace distance
    """
    assert len(basis_d) == len(basis_e)
    start = time.time()
    num_basis = len(basis_d)
    # generate orthogonal basis
    D = np.array(basis_d)
    E = np.array(basis_e)
    
    # the big left matrix
    tmp = np.zeros((2*num_basis, 2*num_basis))
    tmp[:num_basis, :num_basis] = np.dot(D, D.T)
    tmp[:num_basis, num_basis:] = -np.dot(D, E.T)
    tmp[num_basis:, :num_basis] = np.dot(E, D.T)
    tmp[num_basis:, num_basis:] = -np.dot(E, E.T)
    
    # calculate alpha, beta
    alpha_beta = np.dot(
            np.dot(np.linalg.inv(tmp), np.concatenate([D, E], axis=0)), 
            e-d)
    
    # calculate x_star, y_star
    x_star = d + np.dot(alpha_beta[:num_basis], basis_d)
    y_star = e + np.dot(alpha_beta[num_basis:], basis_e)
    
    dist = np.linalg.norm(x_star - y_star)
    return dist, time.time() - start

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

def process_lines(chunk_info_list, pair_list, folder, i):
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
                d, basis_d = load_enrolled_file('{}/{}.npy'.format(folder, file1))
                e, basis_e = load_enrolled_file('{}/{}.npy'.format(folder, file2))
                dist, duration = dist_s_to_s(d, basis_d, e, basis_e)
                score = (2 - dist**2)/2
                score = min(max(score, -1), 1)

                score_list.append((file1, file2, score))
                durations_list.append((file1, file2, duration))

    partid_lineinfo_map[i] = [score_list, durations_list]

    return partid_lineinfo_map

def main(folder, pair_list, score_list):
    # load pair_file
    with open(pair_list, 'r') as f:
        lines = f.readlines()

    fw = open(score_list, 'w')

    print('[ASE] Decrypting features...')    
    start = time.time()
    duration_plain = []    

    n = len(lines)
    if n < 100000:
        for i, line in enumerate(lines):
            file1, file2, _ = line.strip().split(' ')
            # load files
            d, basis_d = load_enrolled_file('{}/{}.npy'.format(folder, file1))
            e, basis_e = load_enrolled_file('{}/{}.npy'.format(folder, file2))

            dist, duration = dist_s_to_s(d, basis_d, e, basis_e)
            # measure time
            score = (2 - dist**2)/2
            score = min(max(score, -1), 1)
            duration_plain.append(duration)
            fw.write('{} {} {}\n'.format(file1, file2, score))
            if i % 1000 == 0:
                print('{}/{}'.format(i, n))
    else:
        # Paralel Generate the scores.
        chunk_info_list =list(chunkify(pair_list,1024*1024))
        lnum = len(chunk_info_list)
        num_jobs = min(lnum, 10)
        idxs = list(range(0,lnum, math.ceil(lnum/num_jobs)))
        idxs.append(lnum)
        # recheck the number of jobs.
        num_jobs = len(idxs) - 1
        result_list = Parallel(n_jobs=num_jobs, verbose=100)(delayed(process_lines)(
            chunk_info_list[idxs[i]:idxs[i+1]], pair_list, folder, i) for i in range(num_jobs))

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
                _, _, duration = durations_list[lineid]
                duration_plain.append(duration)
                if i % 1000 == 0:
                    print('{}/{}'.format(i, n))
                i += 1
    fw.close()
    
    duration = time.time() - start
    print('total duration {}, ase duration {}, calculate {} pairs.\n'.format(duration, sum(duration_plain), n))    


if __name__ == '__main__':        
    main(args.folder, args.pair_list, args.score_list)
