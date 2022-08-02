## SecureVector

A official implementation of SecureVector [Towards Privacy-Preserving, Real-Time and Lossless Feature Matching](https://arxiv.org/abs/2208.00214) and involved baselines of template protection.


### Usage
1. Download data for  lfw/cfp/agedb from [Gdrive](https://drive.google.com/file/d/1iwDNUw6e1dOBaTTheBlXjedB67D_ThMj/view?usp=sharing) or [BaiduDrive](https://pan.baidu.com/s/1vk5lV8m-fgIxvQf3q8ophA?pwd=q58b).

2. Download IJB from BaiduDrive [part1](https://pan.baidu.com/s/1ykRaraO4PTmyMigoj2qNZA?pwd=zcja) and [part2](https://pan.baidu.com/s/1JJiUIXdB0tsyeY81usO2fw?pwd=nasq). Merge them by command `cat data2a* > data2.tar`.

3. Extract them in the root directory. You should have the following structure: 

    **Note**: Features are extracted by [MagFace](https://github.com/IrvingMeng/MagFace). Replace the feat.list if you use another model.
```
    data/
    ├── agedb
    │ ├── agedb_feat.list
    │ └── pair.list
    ├── cfp
    │ ├── cfp_feat.list
    │ └── pair.list
    ├── ijb
    │ ├── ijbb_feat.list
    │ ├── ijbc_feat.list
    │ └── meta
    │     ├── ijbb_face_tid_mid.txt
    │     ├── ijbb_template_pair_label.txt
    │     ├── ijbc_face_tid_mid.txt
    │     └── ijbc_template_pair_label.txt
    └── lfw
    ├── lfw_feat.list
    └── pair.list
```

4. Run evaluations on the face task by:
```
    # [key] for method
        # 0. baseline
        # 1. SecureVector [1]
        # 2. ase [2]
        # 3. ironmask [3]
        # 4. sfm [4]

    export key=1 

    # LFW/CFP/AgeDB
    eval/eval1.sh $key  

    # IJB
    eval/evalibjx.sh $key
```


### References
[1] Qiang Meng, el al, "Towards Privacy-Preserving, Real-Time and Lossless Feature Matching", arXiv 2022.

[2] Dusmanu, Mihai, et al. "Privacy-preserving image features via adversarial affine subspace embeddings." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.* 2021.

[3] Kim, Sunpill, et al. "Ironmask: Modular architecture for protecting deep face template." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.* 2021.

[4] Boddeti, Vishnu Naresh. "Secure face matching using fully homomorphic encryption." *2018 IEEE 9th International Conference on Biometrics Theory, Applications and Systems (BTAS).* IEEE, 2018.
