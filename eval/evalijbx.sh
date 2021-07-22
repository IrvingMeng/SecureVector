M=$1  
# 0. baseline
# 1. invisibleface

METHOD_LIST=('baseline' 'invisibleface' 'ase' 'ironmask' 'sfm')
METHOD=${METHOD_LIST[$M]}

# cd ../
for BM in 'b' 'c'
do
    # Convert ijbx feature to id-template feature
    FEAT_LIST=/face/irving/eval_feats/magface_iresnet100/ijb${BM}_mf_10_110_0.45_0.8_20.list
    IJBX_BASE_FOLD=/face/hnren/6.invisible/data/e3

    if [ ! -f ${IJBX_BASE_FOLD}/ijb${BM}.feat.list ]; then
        python eval/ijbx_template_feature.py \
            --feat_list ${FEAT_LIST} \
            --base_dir /face/irving/data/IJB_eval/IJB${BM^^} \
            --type ${BM} \
            --template_feature ${IJBX_BASE_FOLD}/ijb${BM}_template_mf_10_110_0.45_0.8_20.list \
            --pair_list ${IJBX_BASE_FOLD}/ijb${BM}.pair.list

    fi
    # use the converted id-template feature file.
    FEAT_LIST=${IJBX_BASE_FOLD}/ijb${BM}_template_mf_10_110_0.45_0.8_20.list
    PAIR_LIST=${IJBX_BASE_FOLD}/ijb${BM}.pair.list
    BASE_FOLD=${IJBX_BASE_FOLD}

    FOLD=${BASE_FOLD}/${METHOD}/${BM}
    SCORE_LIST=${FOLD}/score.list

    if [[ $M == 0 ]]
    then
        # generate similarities
        python3 baseline/gen_sim.py --feat_list ${FEAT_LIST} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}

    elif [[ $M == 1 ]]
    then
        KS=1024
        K=32
        # enrollment
        python3 InvisibleFace/enrollment.py --feat_list ${FEAT_LIST} --key_size ${KS} --K ${K} --folder ${FOLD}
        # generate similarities
        python InvisibleFace/crypto_system.py --key_size ${KS} --K ${K} --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}
    
    elif [[ $M == 2 ]]
    then 
        ASE_DIM=4
        # enrollment
        python3 ASE/enrollment.py --feat_list ${FEAT_LIST} --folder ${FOLD} --ase_dim ${ASE_DIM}
        # generate similarities
        python ASE/gen_sim.py --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}  

    elif [[ $M == 3 ]]
    then 
        ALPHA=16
        # enrollment
        python3 IronMask/enrollment.py --feat_list ${FEAT_LIST} --folder ${FOLD} --alpha ${ALPHA}        
        # generate similarities
        python IronMask/gen_sim.py --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}  --alpha ${ALPHA} --feat_list ${FEAT_LIST}

    elif [[ $M == 4 ]]
    then 
        PRECISION=125        
        # enrollment
        python3 SFM/enrollment.py --feat_list ${FEAT_LIST} --folder ${FOLD} --precision ${PRECISION}               
        # generate similarities
        python SFM/gen_sim.py --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}  --precision ${PRECISION}

    else
        echo 'key error'
    fi
done

for BM in 'b' 'c'
do 
    echo [${METHOD}]: ${BM}
    PAIR_LIST=${IJBX_BASE_FOLD}/ijb${BM}.pair.list
    FOLD=${BASE_FOLD}/${METHOD}/${BM}
    SCORE_LIST=${FOLD}/score.list
    python eval/eval_1vn.py --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}
done    