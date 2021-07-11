"""
Benchmark key generation, encryption and decryption.

"""

import random
import resource
import time
import tenseal as ts
import phe.paillier as paillier

test_size = 1000
nums1 = [random.random() for _ in range(test_size)]
nums2 = [random.random() for _ in range(test_size)]


key_sizes = [128, 256, 512, 1024, 2048]
r_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
for key_size in key_sizes:
    # Setup TenSEAL context
    
    pubkey, prikey = paillier.generate_paillier_keypair(n_length=key_size)

    enc_num1 = []
    enc_num2 = []
    print('start encypting')
    start = time.time()
    for num in nums1:
        enc_num1.append(pubkey.encrypt(num))
    for num in nums2:
        enc_num2.append(pubkey.encrypt(num))
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r_init
    duration = (time.time() - start)        
    print('Paillier {}, duration: {}, memory: {}'.format(key_size, duration, r))


key_sizes = [8192]
for key_size in key_sizes:
    # Setup TenSEAL context
    # r_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=key_size,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
    context.generate_galois_keys()
    context.global_scale = 2**40

    enc_num1 = []
    enc_num2 = []
    print('start encypting')
    start = time.time()
    for num in nums1:
        enc_num1.append(ts.ckks_vector(context, [num]))
    for num in nums2:
        enc_num2.append(ts.ckks_vector(context, [num]))
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r_init
    duration = (time.time() - start)
    print('CKKS, duration: {}, memory: {}'.format(duration, r))

