source /miniconda3/bin/activate
export PYTHONPATH=/reconvla/reconvla:$PYTHONPATH
set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=offline
export WANDB_DIR=./wandb

#You need to change the following parameters
# --data_path training data path, it is important, you need to change the data_path if you want to use different target image path or ins+image arrangement
# --image_folder image path, you need to change the image_folder if you want to use different image path
# --target_image_folder target image path, you need to change the target_image_folder if you want to use different target image path
# --output_dir model save path
# --num_epoch training epoch

torchrun --nproc-per-node=8 --nnodes 1 --node_rank 0 \
    --master_addr="localhost" --master_port="20223" \
    \
    train_vla.py \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /reconvla/reconvla/checkpoints/pretrain-checkpoint \
    --output_dir ./checkpoints/checkpoint \
    --vision_tower ./siglip-so400m-patch14-384 \
    --version qwen_2 \
    --mm_pixel_decoder ./pretrained_vae \
    --reconstruct_image_num 1 \
    --data_path /dataset/task_ABC_D/training_r5.json \
    --image_folder /dataset/task_ABC_D/vla_processed_r5 \
    --target_image_folder /dataset/task_ABC_D/vla_processed_r5 \
    --action_stat ./statistics.yaml \
    \
    --mm_projector_type mlp2x_gelu \
    --mm_inv_projector_type denoiser_vit3x \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 10 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --reconstruct_image False \
    --lazy_preprocess True 
