source activate dpo


DISTRIBUTED_ARGS="--nproc_per_node 4 \
                    --nnodes 1 \
                    --node_rank 0 \
                    --master_addr localhost \
                    --master_port 1234"

    


OUTPUT_PATH="path/to/model"
MODEL_PATH="llama3-8B-ifd-pre"


torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn fa2\
    --model_name_or_path $MODEL_PATH \
    --dataset_dir data_info \
    --dataset openhermes_ifd_pre \
    --template default \
    --finetuning_type full \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --ddp_timeout 9000 \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 1000 \
    --plot_loss \
    --seed 0 \
    --num_train_epochs 1 \
    --learning_rate 7e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --bf16 

