#!/bin/bash
# Instruction fine-tuning stage.
# Resumes from completed code model (epoch 12) and trains on chat-formatted instruction data.
# This teaches the model to follow prompts like "Write a function that..." instead of
# just continuing code.

cd /home/vsingh/code/llm-from-scratch/advanced-code-model

export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0

CHECKPOINT="models/code_model_latest.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Instruction fine-tuning requires the completed code model."
    echo "Run clean-data training (run_resume_clean.sh) to completion first."
    exit 1
fi

echo "=========================================="
echo "INSTRUCTION FINE-TUNING"
echo "=========================================="
echo "Starting from: $CHECKPOINT"
cat "${CHECKPOINT%.pth}.json" 2>/dev/null
echo ""
echo "Instruction data: mixed_instruction_train.npy (chat format)"
echo ""

# Lower LR for fine-tuning; 3 epochs is standard for SFT
python scripts/train.py \
  --stage code \
  --model-size 400m \
  --device rocm \
  --data-file mixed_instruction_train.npy \
  --val-file mixed_instruction_val.npy \
  --checkpoint "$CHECKPOINT" \
  --num-epochs 3 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --use-amp \
  --use-rope \
  --steps-per-epoch 100000 \
  --checkpoint-every-mins 30 \
  2>&1 | tee logs/train_instruction_sft.log

echo ""
echo "Instruction fine-tuning finished at $(date)"
