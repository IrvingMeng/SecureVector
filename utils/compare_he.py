"""
Benchmark key generation, encryption and decryption.
"""

import random
import resource
import time
import tenseal as ts
import phe.paillier as paillier

test_size = 10000
nums1 = [int(random.random()*test_size) for _ in range(test_size)]
nums2 = [int(random.random()*test_size) for _ in range(test_size)]


method = 3

if method == 1:
    # key_sizes = [128, 256, 512, 1024]
    key_size = 1024
    r_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   

    pubkey, prikey = paillier.generate_paillier_keypair(n_length=key_size)
    start = time.time()
    enc_num1 = []
    enc_num2 = []    
    for num in nums1:
        enc_num1.append(pubkey.encrypt(num))
    for num in nums2:
        enc_num2.append(pubkey.encrypt(num))
    duration_en = (time.time() - start) 

    # decrypt    
    start = time.time()
    results = []
    for i in range(test_size):
        result = enc_num1[i] + enc_num2[i]
        results.append(result)
    duration_add = (time.time() - start) 
    
    start = time.time()
    for i in range(test_size):
        prikey.decrypt(results[i])
    duration_de = (time.time() - start) 

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r_init
    print('Paillier {}, encrypt duration: {} ms, add duration: {} ms, decrypt duration: {} ms, memory: {} Mb\n'.format(key_size, 
                                                                                                        duration_en/(2*test_size/1000), 
                                                                                                        duration_add/(test_size/1000),
                                                                                                        duration_de/(test_size/1000),
                                                                                                        r/1000))

elif method == 2:
    r_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # context = ts.context(
    #             ts.SCHEME_TYPE.CKKS,
    #             poly_modulus_degree=8192,
    #             coeff_mod_bit_sizes=[60, 40, 40, 60]
    #         )
    # context.generate_galois_keys()
    # context.global_scale = 2**40
    context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=4096,
                coeff_mod_bit_sizes=[30, 20, 20, 30]
            )
    context.generate_galois_keys()
    context.global_scale = 2**20    

    enc_num1 = []
    enc_num2 = []
    print('start encypting')
    start = time.time()
    for num in nums1:
        enc_num1.append(ts.ckks_vector(context, [num]))
    for num in nums2:
        enc_num2.append(ts.ckks_vector(context, [num]))
    duration_en = (time.time() - start)
    
    # decrypt    
    start = time.time()
    results = []
    for i in range(test_size):        
        result = (enc_num1[i] + enc_num2[i])        
        results.append(result)
    duration_add = (time.time() - start) 

    start = time.time()
    for i in range(test_size):
        results[i].decrypt()
    duration_de = (time.time() - start) 

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r_init    
    print('CKKS, encrypt duration: {} ms, add duration: {} ms, decrypt duration: {} ms, memory: {} Mb\n'.format( 
                                                                                                        duration_en/(2*test_size/1000), 
                                                                                                        duration_add/(test_size/1000),
                                                                                                        duration_de/(test_size/1000),
                                                                                                        r/1000))
    

else:     
    r_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    context = ts.context(
                ts.SCHEME_TYPE.BFV, 4096, 1032193)
    context.generate_galois_keys()
    context.global_scale = 2**40
    # context = ts.context(
    #             ts.SCHEME_TYPE.BFV, 8192, 1032193)
    # context.generate_galois_keys()
    # context.global_scale = 2**40    

    enc_num1 = []
    enc_num2 = []
    print('start encypting')
    start = time.time()
    for num in nums1:
        enc_num1.append(ts.bfv_vector(context, [num]))
    for num in nums2:
        enc_num2.append(ts.bfv_vector(context, [num]))    
    duration_en = (time.time() - start)

    print('BFV , encrypt duration: {}'.format(duration_en))
    # decrypt    
    start = time.time()
    results = []
    for i in range(test_size):        
        result = (enc_num1[i] + enc_num2[i])      
        results.append(result)
    duration_add = (time.time() - start) 
    
    start = time.time()
    for i in range(test_size):
        results[i].decrypt()
    duration_de = (time.time() - start) 
    
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - r_init
    print('BFV, encrypt duration: {} ms, add duration: {} ms, decrypt duration: {} ms, memory: {} Mb\n'.format( 
                                                                                                        duration_en/(2*test_size/1000), 
                                                                                                        duration_add/(test_size/1000),
                                                                                                        duration_de/(test_size/1000),
                                                                                                        r/1000))