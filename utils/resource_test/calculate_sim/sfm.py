#!/usr/bin/env python
import sys
import numpy as np
import argparse
import os
import time
import random
import resource
from itertools import repeat
import tenseal as ts
import tenseal.sealapi as sealapi 


def cipher_zero(ctx, batchenc, encryptor):
    plaintext = sealapi.Plaintext()
    batchenc.encode([0], plaintext)
    # ciphertext
    ciphertext = sealapi.Ciphertext(ctx)
    encryptor.encrypt(plaintext, ciphertext)
    return ciphertext

def calculate_sim(cipher1, cipher2, ctx, batchenc, encryptor, decryptor, evaluator, precision):
    # decrypt 
    start = time.time()
    cipher0 = cipher_zero(ctx, batchenc, encryptor)

    evaluator.multiply_inplace(cipher1, cipher2)
    evaluator.relinearize_inplace(cipher1, relin_key)
    encrypted_result = sealapi.Ciphertext(ctx)
    evaluator.add(cipher1, cipher0, encrypted_result)

    slot_count = batchenc.slot_count()
    row_size = int (slot_count / 2)
    for i in range(int(np.log2(row_size))):
        evaluator.rotate_rows(encrypted_result, pow(2,i), gal_key, cipher1)
        evaluator.add_inplace(encrypted_result, cipher1)
    
    plaintext_result = sealapi.Plaintext()
    decryptor.decrypt(encrypted_result, plaintext_result)
    score = batchenc.decode_int64(plaintext_result)[0]/(precision*precision)        
    return score


folder = '/face/irving/eval_feats/template_protection/sfm/lfw/'
pair_list ='/face/irving/data/ms1m_eval/lfw/pair.list'
precision=125

with open(pair_list, 'r') as f:
    lines = f.readlines()


r_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   


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


public_key = sealapi.PublicKey()
public_key.load(ctx, '/face/irving/eval_feats/template_protection/sfm/public_key')
secret_key = keygen.secret_key()
secret_key.load(ctx, '/face/irving/eval_feats/template_protection/sfm/secret_key')
gal_key = sealapi.GaloisKeys()
gal_key.load(ctx, '/face/irving/eval_feats/template_protection/sfm/gal_key')
relin_key = sealapi.RelinKeys()
relin_key.load(ctx, '/face/irving/eval_feats/template_protection/sfm/relin_key')

encryptor = sealapi.Encryptor(ctx, public_key)
decryptor = sealapi.Decryptor(ctx, secret_key)
evaluator = sealapi.Evaluator(ctx)
batchenc = sealapi.BatchEncoder(ctx)

tmp = []
for i, line in enumerate(lines):
    file1, file2, _ = line.strip().split(' ')
    # load files
    cipher1 = sealapi.Ciphertext(ctx)
    cipher2 = sealapi.Ciphertext(ctx)     
    cipher1.load(ctx, '{}/{}'.format(folder, file1))
    cipher2.load(ctx, '{}/{}'.format(folder, file2))
    tmp.append([cipher1, cipher2])


start = time.time()

n = len(lines)    
for i, line in enumerate(lines):
    cipher1, cipher2 = tmp[i]
    score = calculate_sim(cipher1, cipher2, ctx, batchenc, encryptor, decryptor, evaluator, precision)
    # measure time
    if i % 1000 == 0:
        print('{}/{}'.format(i, n))        


duration = time.time() - start
r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r_init
print('total memory {}, total duration {}, encrypted {} features.\n'.format(r, duration, n))  