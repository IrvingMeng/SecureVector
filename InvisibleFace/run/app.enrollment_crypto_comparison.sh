set -v

KS=1024
K=32
LIST=/face/hnren/6.invisible/data/magface_iresnet100/lfw_mf_10_110_0.45_0.8_20.list
FOLD=/face/hnren/6.invisible/data/experiments/en_${KS}_${K}

cd ..

echo "Enroll Feature .."
python3 enrollment.py --file ${LIST} --key_size ${KS} --K ${K} --folder ${FOLD}

echo "Crypto Comparison .."
python3 crypto_system.py  --file1 ${FOLD}/0.npy --file2 ${FOLD}/1.npy --key_size ${KS} --K ${K}