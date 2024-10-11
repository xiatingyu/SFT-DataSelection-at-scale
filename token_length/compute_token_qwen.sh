

index=0
gpu_num=4
for ((i = 0; i < $gpu_num; i++)); do
    start_index=$((i * 111000))
    end_index=$(((i + 1) * 111000))

    gpu=$((i))

    echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
    ((index++))
    (   
        model='your/model/path/Qwen2-7B'
        CUDA_VISIBLE_DEVICES=$gpu  python compute_token_num.py \
            --instruction_path "your/model/path/wildchat.json" \
            --save_path "your/model/path/token_num/wildchat/split/wildchat_qwen2_${i}.jsonl" \
            --start $start_index --end $end_index \
            --model_path ${model}
        
    ) &
    if (($index % $gpu_num == 0)); then wait; fi
done