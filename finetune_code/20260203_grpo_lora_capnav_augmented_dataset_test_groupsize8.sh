#!/bin/bash

echo "Setting up general"

cd /scr

CONDA_DIR=/scr/anaconda3_xiasu
if [ ! -f "Anaconda3-2024.10-1-Linux-x86_64.sh" ]; then
	wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
fi
if [ ! -d "$CONDA_DIR" ]; then
	bash Anaconda3-2025.06-0-Linux-x86_64.sh -b -p ${CONDA_DIR}
	export PATH=${CONDA_DIR}/bin:${PATH}
fi

# Initialize conda
$CONDA_DIR/bin/conda init
source ~/.bashrc

echo "Creating general environment"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
if [ ! -d "$CONDA_DIR/envs/general" ]; then
	conda create -n general python=3.10 -y
	conda activate general
else
	conda activate general	
fi

pip install "transformers>=4.57.0"
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install accelerate
pip install peft
pip install trl
pip3 install ninja
pip3 install deepspeed
pip3 install datasets
pip install av
pip install wandb
export WANDB_API_KEY=dd4b4df67d7dc3edd61a1578b21b00afd33ac850
export WANDB_PROJECT=capnav
export WANDB_ENTITY=mercury1997xia-university-of-washington
module load cuda/12.9.1
module load gcc/11.2.0
pip install "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
cd ~/Qwen3-VL
# This following part is trying to fix the flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so error based on https://github.com/Dao-AILab/flash-attention/issues/1708

cd polyfill-glibc

source activate base
conda activate general
ninja polyfill-glibc
./polyfill-glibc --target-glibc 2.28 /scr/anaconda3_xiasu/envs/general/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so
cd ..

mkdir /scr/xiasu
mkdir /scr/xiasu/hf_hub_cache
mkdir /scr/xiasu/hf_home
mkdir /scr/xiasu/hf_datasets
mkdir /scr/xiasu/hf_cache

export TRANSFORMERS_CACHE=/scr/xiasu/hf_cache
export HF_HOME=/scr/xiasu/hf_home
export HF_DATASETS_CACHE=/scr/xiasu/hf_datasets
export HF_HUB_CACHE=/scr/xiasu/hf_hub_cache
echo "Setup done"

cd /gscratch/makelab/xia/Qwen3-VL

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
batch_size=4
grad_accum_steps=4

# Training entry point
entry_file=qwen-vl-finetune/qwenvl/train/train_qwen_trl_grpo_capnav.py

# Dataset configuration (replace with public dataset names)
datasets=capnav

# Output configuration
run_name="20260203_capnav-train-split-grpo-1epoch-test1-groupsize8"
output_dir=./output/20260203_capnav-train-split-grpo-1epoch-test1-groupsize8

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
    --bf16 True\
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --video_fps 1 \
    --video_max_frames 64 \
    --video_min_frames 16 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --report_to wandb \
    --max_completion_length 1024 \
    --num_generations 8 \
    --max_prompt_length 2048 \
    --use_peft True"

# Add LoRA checkpoint path if provided
if [ -n "$lora_checkpoint" ]; then
    args="${args} --lora_checkpoint_path ${lora_checkpoint}"
fi

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE}\
         ${entry_file} ${args}