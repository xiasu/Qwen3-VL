#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

# DeepSpeed configuration
deepspeed=./qwen-vl-finetune/scripts/zero3.json

# Model configuration
llm=Qwen/Qwen3-VL-8B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=2e-5
batch_size=1
grad_accum_steps=8

# Training entry point
entry_file=qwen-vl-finetune/qwenvl/train/train_qwen_trl_grpo_capnav.py

# Dataset configuration (replace with public dataset names)
datasets=capnav

# Output configuration
run_name="capnav-train-split-grpo-1epoch-test1"
output_dir=./output/capnav-train-split-grpo-1epoch-test1

# LoRA checkpoint to load (optional - set to path of previous LoRA checkpoint)
# If provided, this checkpoint will be loaded as the starting point for GRPO training
# The output will be saved as LoRA weights that can be loaded with PeftModel.from_pretrained
# Example: lora_checkpoint=./output/capnav-train-split-1epoch/checkpoint-656
# Leave empty to start from base model with new LoRA (requires --lora_enable True)
lora_checkpoint="output/capnav-train-split-1epoch/checkpoint-656"

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --video_fps 1 \
    --video_max_frames 64 \
    --video_min_frames 16 \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --report_to wandb"

# Add LoRA checkpoint path if provided
if [ -n "$lora_checkpoint" ]; then
    args="${args} --lora_checkpoint_path ${lora_checkpoint}"
fi

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE}\
         ${entry_file} ${args}