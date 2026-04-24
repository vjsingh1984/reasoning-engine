#!/bin/bash
# Resume training from current best checkpoint on CLEAN dataset.
# Clean = dedup + low-entropy filter applied (rebuild_chinchilla_clean.py).
# Goal: see if removing duplicate/junk rows improves completion quality.

cd /home/vsingh/code/llm-from-scratch/advanced-code-model

export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0

CHECKPOINT="models/code_model_best.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "=========================================="
echo "RESUME TRAINING — CLEAN DATASET"
echo "=========================================="
echo "Starting from: $CHECKPOINT"
cat "${CHECKPOINT%.pth}.json" 2>/dev/null
echo ""
echo "Clean data: mixed_clean_train.npy (6.03B tokens, 5.89M rows)"
echo ""

python scripts/train.py \
  --stage code \
  --model-size 400m \
  --device rocm \
  --data-file mixed_clean_train.npy \
  --val-file mixed_clean_val.npy \
  --checkpoint "$CHECKPOINT" \
  --num-epochs 12 \
  --batch-size 4 \
  --learning-rate 2e-5 \
  --use-amp \
  --use-rope \
  --steps-per-epoch 100000 \
  --checkpoint-every-mins 30 \
  2>&1 | tee logs/train_clean_resume.log

echo ""
echo "Training finished at $(date)"
