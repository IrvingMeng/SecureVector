## SecureVector

A official implementation of SecureVector *Towards Privacy-Preserving, Real-Time and Lossless Feature Matching* and other plug-in template protection baselines.


### Usage

```
# [index] for method
    # 0. baseline
    # 1. SecureVector
    # 2. ase [1]
    # 3. ironmask [2]
    # 4. sfm [3]

export key=1 

# evaluation on LFW/CFP/AgeDB
eval/eval1.sh $key  

# evalutation on IJB
eval/evalibjx.sh $key
```






[1] Dusmanu, Mihai, et al. "Privacy-preserving image features via adversarial affine subspace embeddings." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.* 2021.

[2] Kim, Sunpill, et al. "Ironmask: Modular architecture for protecting deep face template." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.* 2021.

[3] Boddeti, Vishnu Naresh. "Secure face matching using fully homomorphic encryption." *2018 IEEE 9th International Conference on Biometrics Theory, Applications and Systems (BTAS).* IEEE, 2018.