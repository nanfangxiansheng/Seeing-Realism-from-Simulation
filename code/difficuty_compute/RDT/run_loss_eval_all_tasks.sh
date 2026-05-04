#!/bin/bash
export TMPDIR=./tmp
mkdir -p $TMPDIR

echo "=============================="
echo "🔍 RDT multi-task loss evaluation"
echo "=============================="


# ========= model weights =========
export TEXT_ENCODER_NAME="./t5-v1_1-xxl"
export VISION_ENCODER_NAME="./siglip-so400m-patch14-384"
export RDT_CHECKPOINT="./rdt-1b"

MODEL_CONFIG_DIR="$.model_config"
OUTPUT_DIR="$BASE_DIR/logs"
CONFIG_PATH="./configs/base.yaml"
mkdir -p "$OUTPUT_DIR"
OUTPUT_JSON="$OUTPUT_DIR/rdt_1b_finetune_loss.json"


# ========= tasks list =========
TASKS=(
  place_burger_fries-aloha-agilex_randomized_500-300
  place_shoe-aloha-agilex_randomized_500-300
)


MODEL_CONFIG_PATHS=()
for TASK in "${TASKS[@]}"; do
  MODEL_CONFIG_PATHS+=("$MODEL_CONFIG_DIR/${TASK}.yml")
done

# ========= logs =========
LOG_FILE="$OUTPUT_DIR/loss_eval_all_tasks_$(date +'%Y%m%d_%H%M%S').log"
echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] Output JSON: $OUTPUT_JSON"
cd "$BASE_DIR"

python -m train.loss_eval_all_tasks_epoch \
  --config_path "$CONFIG_PATH" \
  --model_config_paths "${MODEL_CONFIG_PATHS[@]}" \
  --checkpoint_path "$RDT_CHECKPOINT" \
  --pretrained_text_encoder_name_or_path "$TEXT_ENCODER_NAME" \
  --pretrained_vision_encoder_name_or_path "$VISION_ENCODER_NAME" \
  --dataset_type finetune \
  --batch_size 8 \
  --num_workers 0 \
  --mixed_precision bf16 \
  --load_from_hdf5 \
  --output_json "$OUTPUT_JSON" \
  > >(tee -a "$LOG_FILE") 2>&1

echo
echo "✅ All tasks finished"
echo "📄 Results saved to: $OUTPUT_JSON"