#!/bin/bash

# Set the master address to localhost for single node
export MASTER_ADDR="127.0.0.1"
export CURRENT_RANK=0

# Since it's single node, we don't need worker_list or SLURM_JOB_NODELIST
n_node=1

echo "MASTER_ADDR="$MASTER_ADDR
echo "Single node setup, no SLURM required."

# OUTPUT of stage 1 script
STAGE1_PATH=$1
# for example, llava-v1.5-7b-mm-align

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
OUTPUT_DIR=/data3/fengjie/model_zoo/citygptv-20241007-mix-v2
mkdir $OUTPUT_DIR
CODE_PATH=/data1/fengjie/CityGPTV/train/VILA/llava/train/train_mem.py
DATA_MIX=llava_instruct+sharegpt4v_gpt4_100k+citygptv_multi+citygptv_single+citygptv_text2img2text+citygptv_img2text2img+citygptv_citywalk_vison+citygpt_citywalk+citygpt_cityqa
# DATA_MIX=citygpt_citywalk+citygpt_cityqa
MODEL_MAX_LENGTH=2048
bs=8  # Adjust batch size as needed for your single GPU
echo "number of nodes:" $n_node7n
echo "per device batch size:" $bs
echo "node rank:" $CURRENT_RANK
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)

torchrun --nnodes=$n_node --nproc_per_node=$NUM_GPUS --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    $CODE_PATH \
    --deepspeed ./zero3.json \
    --model_name_or_path /data3/fengjie/init_ckpt/Llama-3-VILA1.5-8B \
    --version llama_3 \
    --data_mixture $DATA_MIX \
    --vision_tower /data3/fengjie/init_ckpt/siglip-so400m-patch14-384  \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to tensorboard

script_name=$(basename "$0")
cp $script_name $OUTPUT_DIR/$script_name
cp /data1/fengjie/CityGPTV/train/VILA/llava/data/datasets_mixture.py $OUTPUT_DIR/datasets_mixture.py
