M=$1  

METHOD_LIST=('baseline' 'securevector' 'ase' 'ironmask' 'sfm')
METHOD=${METHOD_LIST[$M]}


for BM in 'lfw' #'cfp' 'agedb'
do 
    FEAT_LIST=data/${BM}/${BM}_feat.list
    PAIR_LIST=data/${BM}/pair.list
    BASE_FOLD=results/

    FOLD=${BASE_FOLD}/${METHOD}/${BM}
    SCORE_LIST=${FOLD}/score.list

    if [[ $M == 0 ]]
    then
        # generate similarities
        python3 libs/baseline/gen_sim.py --feat_list ${FEAT_LIST} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}

    elif [[ $M == 1 ]]
    then
        KS=512
        K=64
        # enrollment
        if [ ! -f libs/SecureVector/keys/privatekey_{KS} ]; then
            echo 'generate paillier keys...'
            mkdir libs/SecureVector/keys/
            python libs/SecureVector/crypto_system.py --genkey 1 --key_size ${KS} 
        fi
        python3 libs/SecureVector/enrollment.py --feat_list ${FEAT_LIST} --key_size ${KS} --K ${K} --folder ${FOLD}
        # generate similarities
        python libs/SecureVector/crypto_system.py --key_size ${KS} --K ${K} --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}
    
    elif [[ $M == 2 ]]
    then 
        ASE_DIM=4
        # enrollment
        python3 libs/ASE/enrollment.py --feat_list ${FEAT_LIST} --folder ${FOLD} --ase_dim ${ASE_DIM}
        # generate similarities
        python libs/ASE/gen_sim.py --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}  

    elif [[ $M == 3 ]]
    then 
        ALPHA=16
        # enrollment
        python3 libs/IronMask/enrollment.py --feat_list ${FEAT_LIST} --folder ${FOLD} --alpha ${ALPHA}        
        # generate similarities
        python libs/IronMask/gen_sim.py --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}  --alpha ${ALPHA} --feat_list ${FEAT_LIST}

    elif [[ $M == 4 ]]
    then 
        PRECISION=125        
        if [ ! -f libs/SFM/keys/gal_key ]; then
            echo 'generate SFM keys...'
            mkdir libs/SFM/keys/
            python libs/SFM/gen_sim.py  --genkey 1
        fi 
         # enrollment       
        python3 libs/SFM/enrollment.py --feat_list ${FEAT_LIST} --folder ${FOLD} --precision ${PRECISION}
        # generate similarities
        python libs/SFM/gen_sim.py --folder ${FOLD} --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}  --precision ${PRECISION}

    else
        echo 'key error'
    fi
done

for BM in 'lfw' 'cfp' 'agedb'
do 
    echo [${METHOD}]: ${BM}
    PAIR_LIST=data/${BM}/pair.list
    FOLD=${BASE_FOLD}/${METHOD}/${BM}
    SCORE_LIST=${FOLD}/score.list
    
    # eval for lfw/cfp/agedb
    python eval/eval_1v1.py --pair_list ${PAIR_LIST} --score_list ${SCORE_LIST}
done    