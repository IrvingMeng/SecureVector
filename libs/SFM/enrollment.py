#!/usr/bin/env python
import sys
import numpy as np
import argparse
import os
import time
import random
from itertools import repeat
import shutil
import resource
import tenseal as ts
import tenseal.sealapi as sealapi

# parse the args
parser = argparse.ArgumentParser(description='Enrollment in SecureFaceMatching')
parser.add_argument('--feat_list', type=str)
parser.add_argument('--folder', type=str, help='use to store the keys and encrypted features')
parser.add_argument('--public_key', default='libs/SFM/keys/public_key', type=str, help='path to the public key')
parser.add_argument('--precision', default=125, type=int)
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

def enroll(feature, precision, encryptor, batchenc, ctx):
    """
    enroll a feature
    """
    start = time.time()
    quant_feature = [int(precision*e) for e in feature]
    # plaintext
    plaintext = sealapi.Plaintext()
    batchenc.encode(quant_feature, plaintext)
    # cipertext
    ciphertext = sealapi.Ciphertext(ctx)
    encryptor.encrypt(plaintext, ciphertext)
    duration = time.time() - start
    return ciphertext, duration 


def load_key(public_key):
    poly_modulus_degree = 4096
    plain_modulus = 1032193
    # Setup TenSEAL context
    parms = sealapi.EncryptionParameters(sealapi.SCHEME_TYPE.BFV)
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_plain_modulus(plain_modulus)
    coeff = sealapi.CoeffModulus.BFVDefault(poly_modulus_degree, sealapi.SEC_LEVEL_TYPE.TC128)
    parms.set_coeff_modulus(coeff)
    ctx = sealapi.SEALContext(parms, True, sealapi.SEC_LEVEL_TYPE.TC128)
    keygen = sealapi.KeyGenerator(ctx) 
    pub_key = sealapi.PublicKey()
    pub_key.load(ctx, public_key)

    encryptor = sealapi.Encryptor(ctx, pub_key)
    batchenc = sealapi.BatchEncoder(ctx)
    return encryptor, batchenc, ctx   


def main(feature_list, folder, precision, public_key):
    """
    enrollment in sfm
    """
    # print('loading features...')
    features = load_features(feature_list)
    n, dim = len(features), len(features[0])

    print('[SFM] Encrypting features...')   
    encryptor, batchenc, ctx = load_key(public_key)
    
    start = time.time()
    duration_sfm = []    
    # r_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for i, feature in enumerate(features):        
        result, duration = enroll(feature, precision, encryptor, batchenc, ctx)
        result.save('{}/{}'.format(folder, i))        
        # measure time
        duration_sfm.append(duration)        
        if i % 1000 == 0:
            print('{}/{}'.format(i, n))    
    duration = time.time() - start
    print('total duration {}, sfm duration {}, encrypted {} features.\n'.format(duration, sum(duration_sfm), n))


if __name__ == '__main__':
    if os.path.exists(args.folder):
        shutil.rmtree(args.folder)
    os.makedirs(args.folder)

    main(args.feat_list, args.folder, args.precision, args.public_key)

