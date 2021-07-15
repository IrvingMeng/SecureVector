# benchmark test with encrypto features & comparison.
# LFW, CFP, AGEDB, IJBB, IJBC

source color.sh

# Check KS Yes/No [Yes]
# Check K Yes/No [Yes]
KS=1024
K=32
FOLD=/face/hnren/6.invisible/data/experiments/en_${KS}_${K}

if [ ! -d ${FOLD} ]; then
    mkdir -p ${FOLD}
fi

for benchmark in 'lfw' 'cfp' 'agedb'
do
    gprint "Evaluate .. ${benchmark}"
    FEATURE_FILE=/face/hnren/6.invisible/data/magface_iresnet100/${benchmark}_mf_10_110_0.45_0.8_20.list
    python3 ../eval/crypto_eval_1v1.py \
        --feat_list FEATURE_FILE \
        --pair_list /face/irving/data/ms1m_eval/${benchmark}/pair.list
done


for benchmark in 'b' 'c'
do
    gprint "Evaluate .. ${benchmark}"
    FEATURE_FILE=/face/hnren/6.invisible/data/magface_iresnet100/${benchmark}_mf_10_110_0.45_0.8_20.list
    python3 ../eval/crypto_eval_1vn.py \
        --feat_list FEATURE_FILE \
        --base_dir /face/irving/data/IJB_eval/IJB${benchmark^^}/ \
        --type ${benchmark} 
done