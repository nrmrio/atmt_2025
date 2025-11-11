#!/usr/bin/env bash
set -euo pipefail

# Allow override from environment
RAW_DIR="${RAW_DIR:-cz-en/data/raw}"
DATA_DIR=cz-en/data/prepared
TOK_DIR=cz-en/tokenizers
LOG_DIR=cz-en/logs
OUT_DIR=cz-en/outputs

BATCH_SIZE=128
ACCUM_STEPS=1
LR=5e-4
EPOCHS=3
AMP=bf16
WARMUP_STEPS=4000

mkdir -p "$DATA_DIR" "$TOK_DIR" "$LOG_DIR" "$OUT_DIR" \
         cz-en/checkpoints_nowarm cz-en/checkpoints_warmconst cz-en/checkpoints_warmlin

echo "== PREPROCESS =="
python preprocess.py \
  --source-lang cz \
  --target-lang en \
  --raw-data "$RAW_DIR" \
  --dest-dir "$DATA_DIR" \
  --model-dir "$TOK_DIR" \
  --test-prefix test \
  --train-prefix train \
  --valid-prefix valid \
  --src-vocab-size 8000 \
  --tgt-vocab-size 8000 \
  --src-model "$TOK_DIR/cz-bpe-8000.model" \
  --tgt-model "$TOK_DIR/en-bpe-8000.model" \
  --ignore-existing \
  --force-train

COMMON="--cuda \
--data $DATA_DIR \
--src-tokenizer $TOK_DIR/cz-bpe-8000.model \
--tgt-tokenizer $TOK_DIR/en-bpe-8000.model \
--source-lang cz --target-lang en \
--arch transformer \
--max-epoch $EPOCHS \
--ignore-checkpoints \
--encoder-dropout 0.1 --decoder-dropout 0.1 \
--dim-embedding 256 \
--attention-heads 4 \
--dim-feedforward-encoder 1024 \
--dim-feedforward-decoder 1024 \
--max-seq-len 300 \
--n-encoder-layers 3 \
--n-decoder-layers 3 \
--num-workers 4 --pin-memory \
--batch-size $BATCH_SIZE --accum-steps $ACCUM_STEPS \
--amp $AMP \
--lr $LR"

echo "== TRAIN A) baseline (no warmup) =="
python -u train.py $COMMON \
  --save-dir cz-en/checkpoints_nowarm \
  --log-file $LOG_DIR/nowarm.log \
  --lr-warmup none

echo "== TRAIN B) constant warmup =="
python -u train.py $COMMON \
  --save-dir cz-en/checkpoints_warmconst \
  --log-file $LOG_DIR/warmconst.log \
  --lr-warmup constant --warmup-steps $WARMUP_STEPS

echo "== TRAIN C) linear warmup =="
python -u train.py $COMMON \
  --save-dir cz-en/checkpoints_warmlin \
  --log-file $LOG_DIR/warmlin.log \
  --lr-warmup linear --warmup-steps $WARMUP_STEPS

echo "== TRANSLATE + BLEU =="
for RUN in nowarm warmconst warmlin; do
  CKPT=cz-en/checkpoints_${RUN}/checkpoint_best.pt
  OUT=$OUT_DIR/test_${RUN}.txt
  echo "-- $RUN"
  python -u translate.py \
    --cuda \
    --input $RAW_DIR/test.cz \
    --src-tokenizer $TOK_DIR/cz-bpe-8000.model \
    --tgt-tokenizer $TOK_DIR/en-bpe-8000.model \
    --checkpoint-path $CKPT \
    --batch-size 1 \
    --max-len 300 \
    --output $OUT \
    --bleu \
    --reference $RAW_DIR/test.en | tee $LOG_DIR/bleu_${RUN}.txt
done

echo "== REPORT =="
python -u report_gen.py
echo "== DONE =="
