#!/usr/bin/env python
import sys
import numpy as np
import argparse
import os
import time
import shutil
from numpy.random import default_rng

rng = default_rng()

# parse the args
parser = argparse.ArgumentParser(description='Enrollment in ASE')
parser.add_argument('--feat_list', type=str)
parser.add_argument('--folder', type=str, help='use to store the keys and encrypted features')
parser.add_argument('--ase_dim', type=int, default=4)
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


def gen_random_basis(n=1, dim=512):
    """
    generate n [-1, 1]^dim basis
    """
    basis = []
    for i in range(n):
        base = rng.choice([-1,1], size=dim, replace=True)
        basis.append(base)
    return basis


def gen_adversarial_basis(cand, d, n=1):
    """
    select n basis from candidates
        cand: a list of features
        d:    the translation vector
    """
    total_num = len(cand)
    assert total_num > n
    
    rng = default_rng()
    chosen_idxes = rng.choice(total_num, size=n, replace=False)
    
    return [cand[idx] - d for idx in chosen_idxes]


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


def check_valid(basis):
    """
    check if all basis are linearly independent.
    """
    u_list = []
    for base in basis:
        proj_base = base
        for u in u_list:
            proj_base = proj_base - np.dot(proj_base, u)*u
        if np.linalg.norm(proj_base) < 1e-10:
            return 0
    return 1


def generate_subspace(d, ase_dim, adv_features):
    """
    generate a subspace on d
    """
    # generate source basis
    start = time.time()
    rand_dim = int(ase_dim/2)
    adv_dim = ase_dim - rand_dim
    while 1:
        basis = gen_random_basis(n=rand_dim) + gen_adversarial_basis(adv_features, d, adv_dim)
        if check_valid(basis) == 1:
            break
    
    # permute translation vector
    e = gen_random_basis(1)
    d_1 = ortho_proj(e, d, basis)
    
    # permute basis
    basis_1 = []
    for i in range(ase_dim):
        e = gen_random_basis(1)
        proj_e = ortho_proj(e, d, basis)
        base_1 = proj_e - d_1
        basis_1.append(base_1)
    return [d_1, basis_1], time.time() - start


def main(feature_list, folder, ase_dim):
    """
    enrollment
    """
    # print('loading features...')
    features = load_features(feature_list)
    n, dim = len(features), len(features[0])
    # L_list = [i for i in range(0, 2*L)]

    print('[ASE] Encrypting features...')
    start = time.time()
    duration_plain = []
    for i, feature in enumerate(features):
        ase_result, duration = generate_subspace(feature, ase_dim, features)
        np.save('{}/{}.npy'.format(folder, i), np.array(ase_result, np.dtype(object)))
        # measure time
        duration_plain.append(duration)
        if i % 1000 == 0:
            print('{}/{}'.format(i, n))
    duration = time.time() - start
    print('total duration {}, ase duration {},  encrypted {} features.\n'.format(duration, sum(duration_plain), n))


if __name__ == '__main__':
    if os.path.exists(args.folder):
        shutil.rmtree(args.folder)
    os.makedirs(args.folder)

    main( args.feat_list, args.folder, args.ase_dim)

