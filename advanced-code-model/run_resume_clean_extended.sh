#!/bin/bash
# Continue clean training beyond epoch 12.
# Resumes from latest checkpoint and trains to epoch 15.
# Goal: more clean-data epochs to unlearn repetition patterns.

cd /home/vsingh/code/llm-from-scratch/advanced-code-model

export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0

# Use periodic checkpoint — it has full scheduler state
# End-of-epoch checkpoints don't save scheduler state, causing LR reset
CHECKPOINT="models/code_model_periodic.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Periodic checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found in models/"
    exit 1
fi

echo "=========================================="
echo "EXTENDED CLEAN TRAINING (epochs 12→15)"
echo "=========================================="
echo "Resuming from: $CHECKPOINT ($(date -d @$NEWEST_TIME))"
cat "${CHECKPOINT%.pth}.json" 2>/dev/null || echo "  (no metadata file)"
echo ""
echo "Clean data: mixed_clean_train.npy (6.03B tokens)"
echo "Target: 3 more epochs (12→15)"
echo ""

# Continue to epoch 15 (training script auto-skips completed epochs)
python scripts/train.py \
  --stage code \
  --model-size 400m \
  --device rocm \
  --data-file mixed_clean_train.npy \
  --val-file mixed_clean_val.npy \
  --checkpoint "$CHECKPOINT" \
  --num-epochs 15 \
  --batch-size 4 \
  --learning-rate 2e-5 \
  --use-amp \
  --use-rope \
  --steps-per-epoch 100000 \
  --checkpoint-every-mins 30 \
  2>&1 | tee logs/train_clean_extended.log

echo ""
echo "Extended training finished at $(date)"
