
index=0
gpu_num=8
for ((i = 4; i < $gpu_num; i++)); do
    start_index=$((i * 55300))
    end_index=$(((i + 1) * 55300))

    gpu=$((i))
    echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
    ((index++))
    (
        model="path/to/Qwen2-1.5B"
        CUDA_VISIBLE_DEVICES=$gpu python self_reflection/sentence_level.py \
                --start $start_index --end $end_index \
                --model_name_or_path $model \
                --rating_prompt_file "data/rating_prompt.txt" \
                --input_file "/data/wildchat-1M-en.json" \
                --output_file "wild/qwen2/1.5/sentence_${i}.json" \
                --k 5 \
                --proportion 0.2 \
                --alpha 0.2 
        sleep 1
        model="path/to/Qwen2-7B"
        CUDA_VISIBLE_DEVICES=$gpu python self_reflection/sentence_level.py \
                --start $start_index --end $end_index \
                --model_name_or_path $model \
                --rating_prompt_file "data/rating_prompt.txt" \
                --input_file "data/wildchat-1M-en.json" \
                --output_file "wild/qwen2/7/sentence_${i}.json" \
                --k 5 \
                --proportion 0.2 \
                --alpha 0.2 

    ) &
    if (($index % $gpu_num == 0)); then wait; fi
done 