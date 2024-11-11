train_file="path/to/WildChat_val.json" # path to data
model="path/to/out/llama3-8B-Wildchat-lora-seed3/checkpoint-1" # path to model
dims=8192 # dimension of projection, can be a list
output_path="path/to/wildchat/val/llama3-8B-ckpt1" # path to output

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

CUDA_VISIBLE_DEVICES=0 python3 -m less.data_selection.get_info \
    --info_type grads \
    --model_path $model \
    --output_path $output_path \
    --gradient_projection_dimension $dims \
    --gradient_type sgd \
    --train_file $train_file 


train_file="path/to/WildChat_val.json" # path to data
model="path/to/out/llama3-8B-Wildchat-lora-seed3/checkpoint-2" # path to model
dims=8192 # dimension of projection, can be a list
output_path="path/to/wildchat/val/llama3-8B-ckpt2" # path to output


if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi


CUDA_VISIBLE_DEVICES=0 python3 -m less.data_selection.get_info \
    --info_type grads \
    --model_path $model \
    --output_path $output_path \
    --gradient_projection_dimension $dims \
    --gradient_type sgd \
    --train_file $train_file 

train_file="path/to/WildChat_val.json" # path to data
model="path/to/out/llama3-8B-Wildchat-lora-seed3/checkpoint-3" # path to model
dims=8192 # dimension of projection, can be a list
output_path="path/to/wildchat/val/llama3-8B-ckpt3" # path to output


if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi


CUDA_VISIBLE_DEVICES=0 python3 -m less.data_selection.get_info \
    --info_type grads \
    --model_path $model \
    --output_path $output_path \
    --gradient_projection_dimension $dims \
    --gradient_type sgd \
    --train_file $train_file 


train_file="path/to/WildChat_val.json" # path to data
model="path/to/out/llama3-8B-Wildchat-lora-seed3/checkpoint-4" # path to model
dims=8192 # dimension of projection, can be a list
output_path="path/to/wildchat/val/llama3-8B-ckpt4" # path to output

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi


CUDA_VISIBLE_DEVICES=0 python3 -m less.data_selection.get_info \
    --info_type grads \
    --model_path $model \
    --output_path $output_path \
    --gradient_projection_dimension $dims \
    --gradient_type sgd \
    --train_file $train_file 