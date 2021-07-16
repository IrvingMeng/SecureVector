# this script aims to convert ijbx features to template features.
#!/usr/bin/env bash
source argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('--feat_list', type=str)
parser.add_argument('--base_dir', type=str)
parser.add_argument('--type', type=str,
                    default='c')
parser.add_argument('--embedding_size', type=int,
                    default=512)
parser.add_argument('--eval1v1_feature', type=str)
parser.add_argument('--eval1v1_pair_list', type=str)
EOF

# python ijbx_template_feature.py \
#     --feat_list ${FEAT_LIST} \
#     --base_dir ${BASE_DIR} \
#     --type ${TYPE} \
#     --embedding_size ${EMBEDDING_SIZE} \
#     --template_feature ${EVAL1V1_FEATURE} \
#     --pair_list ${EVAL1V1_PAIR_LIST}

for part in {1..79}
do
echo "cat ${EVAL1V1_FEATURE}_part${part} >> ${EVAL1V1_FEATURE}"
cat ${EVAL1V1_FEATURE}_part${part} >> ${EVAL1V1_FEATURE}
echo "cat ${EVAL1V1_PAIR_LIST}_part${part} >> ${EVAL1V1_PAIR_LIST}"
cat ${EVAL1V1_PAIR_LIST}_part${part} >> ${EVAL1V1_PAIR_LIST}
# rm ${EVAL1V1_FEATURE}_part${part}
# rm ${EVAL1V1_PAIR_LIST}_part${part}
done