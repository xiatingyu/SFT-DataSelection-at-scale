
index=0
gpu_num=8
for ((i = 4; i < $gpu_num; i++)); do
    start_index=$((i * 55300))
    end_index=$(((i + 1) * 55300))

    gpu=$((i-4))
    echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
    ((index++))
    (
        model="path/to/llama3-8b-base"
        CUDA_VISIBLE_DEVICES=$gpu python self_reflection/sentence_level.py \
                --start $start_index --end $end_index \
                --model_name_or_path $model \
                --rating_prompt_file "data/rating_prompt.txt" \
                --input_file "data/wildchat.json" \
                --output_file "wild/llama/sentence_${i}.json" \
                --k 5 \
                --proportion 0.2 \
                --alpha 0.2 
       

    ) &
    if (($index % $gpu_num == 0)); then wait; fi
done 