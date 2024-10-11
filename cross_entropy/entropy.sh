index=0
gpu_num=8
for ((i = 0; i < $gpu_num; i++)); do
    start_index=$((i * 125200))
    end_index=$(((i + 1) * 125200))

    gpu=$((i))
    echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
    ((index++))
    (
        model="path/to/Qwen2-7B"
        CUDA_VISIBLE_DEVICES=$gpu python entropy_response.py \
                --start $start_index --end $end_index \
                --base_model $model \
                --data_file "data/openhermes.json" \
                --output_file "cross_entropy/openhermes/split/entropy_qwen2_${i}.jsonl" \
         

    ) &
    if (($index % $gpu_num == 0)); then wait; fi
done 

