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


precision = 125
public_key = '/face/irving/eval_feats/template_protection/sfm/public_key'
feature_list = '/face/irving/eval_feats/magface_iresnet100/lfw_mf_10_110_0.45_0.8_20.list'


# print('loading features...')
features = load_features(feature_list)
n, dim = len(features), len(features[0])

r_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
start = time.time()
print('[SFM] Encrypting features...') 

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

plaintext_list = [sealapi.Plaintext() for _ in range(n)]
for i, feature in enumerate(features):  
    quant_feature = [int(precision*e) for e in feature]
    # plaintext
    plaintext = plaintext_list[i]
    batchenc.encode(quant_feature, plaintext)
    # cipertext
    ciphertext = sealapi.Ciphertext(ctx)
    encryptor.encrypt(plaintext, ciphertext)                        
    if i % 1000 == 0:
        print('{}/{}'.format(i, n))    
duration = time.time() - start
r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r_init
print('total memory {}, total duration {}, encrypted {} features.\n'.format(r, duration, n))

