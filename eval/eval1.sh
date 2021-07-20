M=$1  
# 0. baseline
# 1. invisibleface

METHOD_LIST=('baseline' 'invisibleface' 'ase' 'ironmask' 'sfm')
METHOD=${METHOD_LIST[$M]}

# cd ../

for BM in 'lfw' 'cfp' 'agedb'
do 
    FEAT_LIST=/face/irving/eval_feats/magface_iresnet100/${BM}_mf_10_110_0.45_0.8_20.list
    PAIR_LIST=/face/irving/data/ms1m_eval/${BM}/pair.list
    BASE_FOLD=/face/irving/eval_feats/template_protection

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

for BM in 'lfw' 'cfp' 'agedb'
do 
    echo [${METHOD}]: ${BM}
    PAIR_LIST=/face/irving/data/ms1m_eval/${BM}/pair.list
    FOLD=${BASE_FOLD}/${METHOD}/${BM}
    SCORE_LIST=${FOLD}/score.list
    
    # eval for lfw/cfp/agedb
    python eval/eval_1v1.py --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}    
done    