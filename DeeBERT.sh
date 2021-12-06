PATH_TO_DATA="glue_data"
MODEL_TYPE="bert"  # bert or roberta
MODEL_SIZE="base"  # base or large
# CoLA  acceptability
# SST-2 sentiment
# MRPC  paraphrase
# STS-B sentence similarity doesn't work
# QQP   paraphase
# MNLI  NLI
# QNLI  QA/NLI
# RTE   NLI
# WNLI  coreference/NLI

# DATASETS="CoLA SST-2 MRPC STS-B QQP MNLI QNLI RTE WNLI"
DATASETS="MNLI QNLI RTE WNLI"
# DATASETS="RTE"
N_GPU=2

for DATASET in $DATASETS; do
  echo $DATASET
  echo "--------train--------"
  bash scripts/train.sh ${PATH_TO_DATA} ${MODEL_TYPE} ${MODEL_SIZE} ${DATASET} ${N_GPU}
  echo "--------train_highway--------"
  bash scripts/train_highway.sh ${PATH_TO_DATA} ${MODEL_TYPE} ${MODEL_SIZE} ${DATASET} ${N_GPU}
  echo "--------eval--------"
  bash scripts/eval_highway.sh ${PATH_TO_DATA} ${MODEL_TYPE} ${MODEL_SIZE} ${DATASET}
  bash scripts/eval_entropy.sh ${PATH_TO_DATA} ${MODEL_TYPE} ${MODEL_SIZE} ${DATASET}
done
