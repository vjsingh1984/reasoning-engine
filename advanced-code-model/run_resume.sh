#!/bin/bash
# Resume training after WSL2 restart / crash
# Now supports step-level resume from periodic checkpoints.
#
# Checkpoint priority: periodic (mid-epoch) > latest (end-of-epoch) > best
# The script auto-detects which is newest and resumes from there.
# Training skips ahead to the exact epoch+step where it left off.

cd /home/vsingh/code/llm-from-scratch/advanced-code-model

# Workaround for ROCm scratch reclaim assertion on RDNA 4 / WSL2
export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0

# Pick the newest checkpoint
CHECKPOINT=""
NEWEST_TIME=0
for f in models/code_model_periodic.pth models/code_model_latest.pth models/code_model_best.pth; do
    if [ -f "$f" ]; then
        FTIME=$(stat -c %Y "$f" 2>/dev/null || echo 0)
        if [ "$FTIME" -gt "$NEWEST_TIME" ]; then
            NEWEST_TIME=$FTIME
            CHECKPOINT="$f"
        fi
    fi
done

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found in models/"
    exit 1
fi

echo "Resuming from: $CHECKPOINT ($(date -d @$NEWEST_TIME))"
echo "Checkpoint metadata:"
cat "${CHECKPOINT%.pth}.json" 2>/dev/null || echo "  (no metadata file)"
echo ""

# Use --num-epochs 3 (the original total) so epoch indexing is correct.
# The resume logic auto-skips completed epochs and steps.
python scripts/train.py \
  --stage code \
  --model-size 400m \
  --device rocm \
  --data-file mixed_train.npy \
  --val-file mixed_val.npy \
  --checkpoint "$CHECKPOINT" \
  --num-epochs 10 \
  --batch-size 4 \
  --learning-rate 2e-5 \
  --use-amp \
  --use-rope \
  --steps-per-epoch 100000 \
  --checkpoint-every-mins 30 \
  2>&1 | tee logs/train_run7_400m_resume.log

echo "Training finished at $(date)"
