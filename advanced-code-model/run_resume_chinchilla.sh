#!/bin/bash
# Resume training from periodic checkpoint with Chinchilla-optimal dataset.
#
# Checkpoint: epoch 6, step 37,251
# Dataset: 8.00B tokens (Chinchilla-optimal)
# Resume and complete training

cd /home/vsingh/code/llm-from-scratch/advanced-code-model

# ROCm RDNA 4 / WSL2 workaround
export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0

CHECKPOINT="models/code_model_periodic.pth"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "=========================================="
echo"RESUMING TRAINING - CHINCHILLA OPTIMAL"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
cat "${CHECKPOINT%.pth}.json" 2>/dev/null || echo "  (no metadata file)"
echo ""

echo "Dataset info (Chinchilla-optimal 8.00B tokens):"
python3 -c "
import numpy as np
d = np.load('data/processed/mixed_train.npy')
print(f'  Train: {d.shape[0]:,} seqs × {d.shape[1]} = {d.shape[0]*d.shape[1]/1e9:.2f}B tokens')
print(f'  File size: {d.nbytes / 1e9:.1f}GB')
d = np.load('data/processed/mixed_val.npy')
print(f'  Val:   {d.shape[0]:,} seqs × {d.shape[1]} = {d.shape[0]*d.shape[1]/1e9:.2f}B tokens')
print(f'  Bash+Code focus: 81% (6.51B tokens)')
print(f'  Chinchilla optimal: 8.00B tokens (102%)')
"
echo ""

# Resume training - will continue from epoch 6, step 37,251
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
  2>&1 | tee logs/train_chinchilla_optimal.log

echo ""
echo "Training finished at $(date)"
