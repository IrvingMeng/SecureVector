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

# parse the args
parser = argparse.ArgumentParser(description='Match in SecureFaceMatching')
parser.add_argument('--folder', default='', type=str,
                    help='fold which stores the encrypted features')
parser.add_argument('--pair_list', default='', type=str, help='pair file')
parser.add_argument('--score_list', type=str,
                    help='a file which stores the scores')
parser.add_argument('--genkey', default=0, type=int)
parser.add_argument('--precision', default=125, type=int)
args = parser.parse_args()

poly_modulus_degree = 4096
plain_modulus = 1032193
# Setup TenSEAL context
parms = sealapi.EncryptionParameters(sealapi.SCHEME_TYPE.BFV)
parms.set_poly_modulus_degree(poly_modulus_degree)
parms.set_plain_modulus(plain_modulus)
coeff = sealapi.CoeffModulus.BFVDefault(
    poly_modulus_degree, sealapi.SEC_LEVEL_TYPE.TC128)
parms.set_coeff_modulus(coeff)
ctx = sealapi.SEALContext(parms, True, sealapi.SEC_LEVEL_TYPE.TC128)
keygen = sealapi.KeyGenerator(ctx)

if args.genkey == 1:
    # public key
    public_key = sealapi.PublicKey()
    keygen.create_public_key(public_key)
    # secret key
    secret_key = keygen.secret_key()
    # galois keys
    gal_key = sealapi.GaloisKeys()
    keygen.create_galois_keys(gal_key)
    # relin keys
    relin_key = sealapi.RelinKeys()
    keygen.create_relin_keys(relin_key)

    public_key.save('libs/SFM/keys/public_key')
    secret_key.save('libs/SFM/keys/secret_key')
    gal_key.save('libs/SFM/keys/gal_key')
    relin_key.save('libs/SFM/keys/relin_key')
    exit(1)
else:
    public_key = sealapi.PublicKey()
    public_key.load(ctx, 'libs/SFM/keys/public_key')
    secret_key = keygen.secret_key()
    secret_key.load(ctx, 'libs/SFM/keys/secret_key')
    gal_key = sealapi.GaloisKeys()
    gal_key.load(ctx, 'libs/SFM/keys/gal_key')
    relin_key = sealapi.RelinKeys()
    relin_key.load(ctx, 'libs/SFM/keys/relin_key')


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
    row_size = int(slot_count / 2)
    for i in range(int(np.log2(row_size))):
        evaluator.rotate_rows(encrypted_result, pow(2, i), gal_key, cipher1)
        evaluator.add_inplace(encrypted_result, cipher1)

    plaintext_result = sealapi.Plaintext()
    decryptor.decrypt(encrypted_result, plaintext_result)
    score = batchenc.decode_int64(plaintext_result)[0]/(precision*precision)
    return score, time.time() - start


def main(folder, pair_list, score_list, precision):
    # load pair_file
    with open(pair_list, 'r') as f:
        lines = f.readlines()

    fw = open(score_list, 'w')

    print('[SFM] Decrypting features...')
    encryptor = sealapi.Encryptor(ctx, public_key)
    decryptor = sealapi.Decryptor(ctx, secret_key)
    evaluator = sealapi.Evaluator(ctx)
    batchenc = sealapi.BatchEncoder(ctx)

    start = time.time()
    duration_sfm = []

    cipher1 = sealapi.Ciphertext(ctx)
    cipher2 = sealapi.Ciphertext(ctx)

    n = len(lines)
    for i, line in enumerate(lines):
        file1, file2, _ = line.strip().split(' ')
        # load files
        cipher1.load(ctx, '{}/{}'.format(folder, file1))
        cipher2.load(ctx, '{}/{}'.format(folder, file2))
        score, duration = calculate_sim(
            cipher1, cipher2, ctx, batchenc, encryptor, decryptor, evaluator, precision)
        # measure time
        duration_sfm.append(duration)
        fw.write('{} {} {}\n'.format(file1, file2, score))
        if i % 1000 == 0:
            print('{}/{}'.format(i, n))
    fw.close()

    duration = time.time() - start
    print('total duration {}, sfm duration {}, calculate {} pairs.\n'.format(
        duration, sum(duration_sfm), n))


if __name__ == '__main__':
    main(args.folder, args.pair_list, args.score_list, args.precision)
