# task=$1
k=$1

train_file="path/to/wildchat-1M-en.jsonl" 
model="path/to/llama3-8B-Wildchat-lora-seed3/checkpoint-${k}" # path to model
dims=8192 # dimension of projection, can be a list
gradient_type="adam"

# linecount=$(wc -l < "$train_file")
# echo $linecount
# gpu_num=8
# num=$((linecount/gpu_num + 1))
num=25000
echo $num
index=0
gpu_num=4
for ((i = 0; i < $gpu_num; i++)); do
    start_index=$((i * num))
    end_index=$(((i + 1) * num))
    gpu=$((i))
    j=$((i))
    echo $task
    echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
    ((index++))
    (    
        output_path="path/to/llama3-8B-ckpt-${k}/${j}" # path to output

        if [[ ! -d $output_path ]]; then
            mkdir -p $output_path
        fi
        CUDA_VISIBLE_DEVICES=$gpu python3 -m less.data_selection.get_info \
            --train_file $train_file \
            --info_type grads \
            --model_path $model \
            --output_path $output_path \
            --gradient_projection_dimension $dims \
            --gradient_type $gradient_type \
            --start $start_index \
            --end $end_index 
        sleep 1
    ) &
    if (($index % $gpu_num == 0)); then wait; fi
done 
