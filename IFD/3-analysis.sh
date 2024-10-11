
index=0
gpu_num=8
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * 125250))
  end_index=$(((i + 1) * 125250))

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
      model='path/to/Qwen2-7B-ifd-pre'
      CUDA_VISIBLE_DEVICES=$gpu python cherry_seletion/data_analysis.py \
            --model_name_or_path ${model} \
            --start $start_index --end $end_index \
            --data_path "data/OpenHermes2.5.json" \
            --save_path "cherry_data/qwen2-multi-turn/split/OpenHermes_pre_${i}.pt" \
            --max_length 4096 \
            --prompt alpaca \
            --mod cherry
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done

