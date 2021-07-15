# TemplateProtection

A library for plug-in template protection methods.


## InvisibleFace
[To apper in CVPR2022] InvisibleFace: Towards Privacy-preserving, Real-time and  Accurate Template Matching

```
export KS=1024
export K=64

export BM=cfp
export LIST=/face/irving/eval_feats/magface_iresnet100/${BM}_mf_10_110_0.45_0.8_20.list
export PAIR=/face/irving/data/ms1m_eval/${BM}/pair.list
export FOLD=/face/irving/eval_feats/invisibleface/en_${KS}_${K}/${BM}

# enrollment
python3 InvisibleFace/enrollment.py --file ${LIST} --key_size ${KS} --K ${K} --folder ${FOLD}

# generate similarities
python InvisibleFace/crypto_system.py --key_size ${KS} --K ${K} --folder ${FOLD} --pair_file ${PAIR} --score_file ${FOLD}/score.list
```

## FHE
[2018] Secure Face Matching Using Fully Homomorphic Encryption

## HERS

[2021] HERS: Homomorphically Encrypted Representation Search

## IronMask
[CVPR2021] IronMask: Modular Architecture for Protecting Deep Face Template

Note: can only protect gallery. Probe templates are exposed.

## 
[CVPR2021 Best paper candidate] Privacy-Preserving Image Features via Adversarial Affine Subspace Embeddings