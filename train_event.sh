accelerate launch train_svd_event.py \
    --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
    --pretrain_unet=outputs/event_motion_svd \
    --pretrain_controlnext=outputs/event_motion_svd \
    --cache_dir=/root/autodl-tmp/models_cache \
    --base_folder=/root/autodl-tmp/TikTok_event \
    --validation_image_folder=/root/autodl-tmp/TikTok_event/val_set/video_part \
    --validation_control_folder=/root/autodl-tmp/TikTok_event/event_val_set/video_part \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=50000 \
    --width=512 \
    --height=768 \
    --checkpointing_steps=10000 \
    --checkpoints_total_limit=5 \
    --learning_rate=1e-5 \
    --lr_warmup_steps=0 \
    --seed=2024 \
    --mixed_precision="fp16" \
    --validation_steps=100 \
    --num_frames=7
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --use_8bit_adam