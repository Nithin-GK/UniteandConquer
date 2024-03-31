#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --large_size 256  --small_size 256 --learn_sigma True --noise_schedule linear --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
CUDA_VISIBLE_DEVICES="5,6" NCCL_P2P_DISABLE=1  torchrun --nproc_per_node=2 --master_port=4326 scripts/train.py $MODEL_FLAGS