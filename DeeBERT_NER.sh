PATH_TO_DATA="ner_data"
MODEL_TYPE="bert"  # bert or roberta
MODEL_SIZE="base"  # base or large

DATASETS="CoNLL"
N_GPU=4
Ks="1 2 3 4 5 6 7 8 9 10 11 12"

for DATASET in $DATASETS; do
  echo $DATASET
  echo "--------train--------"
  bash scripts/ner/train.sh ${PATH_TO_DATA} ${MODEL_TYPE} ${MODEL_SIZE} ${DATASET} ${N_GPU}
  for K in $Ks; do
    bash scripts/ner/train_KL.sh ${PATH_TO_DATA} ${MODEL_TYPE} ${MODEL_SIZE} ${DATASET} ${N_GPU} ${K}
  done
  echo "--------train_highway--------"
  bash scripts/ner/train_highway.sh ${PATH_TO_DATA} ${MODEL_TYPE} ${MODEL_SIZE} ${DATASET} ${N_GPU}
  echo "--------eval--------"
  bash scripts/ner/eval_highway.sh ${PATH_TO_DATA} ${MODEL_TYPE} ${MODEL_SIZE} ${DATASET}
  bash scripts/ner/eval_entropy.sh ${PATH_TO_DATA} ${MODEL_TYPE} ${MODEL_SIZE} ${DATASET}
done
