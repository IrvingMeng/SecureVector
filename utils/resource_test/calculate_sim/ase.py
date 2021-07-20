#!/usr/bin/env python
import sys
import numpy as np
import argparse
import os
import time
import random
import resource

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
    return dist


folder = '/face/irving/eval_feats/template_protection/ase/lfw/'
pair_list ='/face/irving/data/ms1m_eval/lfw/pair.list'



with open(pair_list, 'r') as f:
    lines = f.readlines()

r_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   

tmp = []
for i, line in enumerate(lines):
    file1, file2, _ = line.strip().split(' ')
    # load files
    d, basis_d = load_enrolled_file('{}/{}.npy'.format(folder, file1))
    e, basis_e = load_enrolled_file('{}/{}.npy'.format(folder, file2))
    tmp.append([d, basis_d, e, basis_e])

print('[ASE] Decrypting features...')    
start = time.time()
n = len(lines)
for i, onepair in enumerate(tmp):
    d, basis_d, e, basis_e = onepair
    dist = dist_s_to_s(d, basis_d, e, basis_e)
    # measure time
    score = (2 - dist**2)/2
    score = min(max(score, -1), 1)
    if i % 1000 == 0:
        print('{}/{}'.format(i, n))        


duration = time.time() - start
r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r_init
print('total memory {}, total duration {}, encrypted {} features.\n'.format(r, duration, n))  


