# #!/bin/bash
# CKPT=105

# TRAINING_DATA_NAME=dolly
# TRAINING_DATA_FILE=../data/train/processed/dolly/dolly_data.jsonl # when changing data name, change the data path accordingly
# GRADIENT_TYPE="adam"
# MODEL_PATH=../out/llama2-7b-p0.05-lora-seed3/checkpoint-${CKPT}
# OUTPUT_PATH=../grads/llama2-7b-p0.05-lora-seed3/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}
# DIMS="8192"

# ./less/scripts/get_info/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"


train_file="/cpfs01/shared/Group-m6/xiatingyu.xty/data/OpenHermes2.5.json" #
model="/cpfs01/shared/Group-m6/xiatingyu.xty/model/Qwen1.5-1.8B-lora/checkpoint-1" # path to model
output_path="tmp/Qwen1.5-1.8B-ckpt1" # path to output
dims=8192 # dimension of projection, can be a list
gradient_type="adam"

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

python3 -m less.data_selection.get_info \
--train_file $train_file \
--info_type grads \
--model_path $model \
--output_path $output_path \
--gradient_projection_dimension $dims \
--gradient_type $gradient_type \
--max_samples 200
