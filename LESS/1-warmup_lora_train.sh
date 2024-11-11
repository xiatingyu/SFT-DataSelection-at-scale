
export WANDB_API_KEY=cac878fde2ea0334527a15036103638df587d233


source less/scripts/train/base_training_args.sh

train_file="/path/to/WildChat_train.json"
model_path="/path/to/llama3-8B"
percentage=1.0
data_seed=3
job_name=llama3-8B-Wildchat-lora-seed3

output_dir=path/to/out/${job_name}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi


training_args="$base_training_args \
--model_name_or_path $model_path \
--fsdp 'full_shard auto_wrap' \
--fsdp_config llama_finetune \
--output_dir $output_dir \
--percentage $percentage \
--data_seed $data_seed \
--train_files $train_file"

eval "$header" "$training_args"
