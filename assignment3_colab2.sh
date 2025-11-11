#!/usr/bin/env bash
set -euo pipefail

# --- directory layout (Colab /content/atmt_2025) ---
RAW_DIR="${RAW_DIR:-cz-en/data/raw}"
DATA_DIR=cz-en/data/prepared
TOK_DIR=cz-en/tokenizers
LOG_DIR=cz-en/logs
OUT_DIR=cz-en/outputs
CKPT_DIR=cz-en/checkpoints_warmlin        # choose any of trained runs
RESTORE_FILE=checkpoint_best.pt           # or checkpoint_last.pt
DECODE_OUT=cz-en/decoding                 # where decoded hypotheses go

mkdir -p "$DECODE_OUT" "$LOG_DIR"

echo "== DECODE-ONLY TASK 2 =="
echo "Using checkpoint from: $CKPT_DIR/$RESTORE_FILE"

python -u train2.py \
  --cuda \
  --data cz-en/data/prepared \
  --source-lang cz --target-lang en \
  --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
  --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
  --save-dir cz-en/checkpoints_warmlin \
  --restore-file checkpoint_best.pt \
  --arch transformer \
  --dim-embedding 256 \
  --dim-feedforward-encoder 1024 \
  --dim-feedforward-decoder 1024 \
  --attention-heads 4 \
  --n-encoder-layers 3 \
  --n-decoder-layers 3 \
  --max-seq-len 300
  --decode-only \
  --output-dir "$DECODE_OUT" \
  --log-file "$LOG_DIR/decoding_task2.log"

echo "== DONE | Results saved in $DECODE_OUT =="
