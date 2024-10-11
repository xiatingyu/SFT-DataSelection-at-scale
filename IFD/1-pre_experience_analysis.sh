source activate sft

index=0
gpu_num=8
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * 55300))
  end_index=$(((i + 1) * 55300))

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
      model='path/to/Qwen2-7B'
      CUDA_VISIBLE_DEVICES=$gpu python cherry_seletion/data_analysis.py \
            --model_name_or_path ${model} \
            --start $start_index --end $end_index \
            --data_path "data/wildchat-1M-en.json" \
            --save_path "analysis/qwen2-multi-turn/split/wildchat_analysis_${i}.pt" \
            --max_length 4096 \
            --prompt alpaca \
            --mod pre
      

  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done

sleep 10
python data_merge.py \
    --pt_data_path "analysis/qwen2-multi-turn/split/wildchat_analysis" \
    --pt_save_path "analysis/qwen2-multi-turn/wildchat_analysis.pt"


pip install scikit-learn
CUDA_VISIBLE_DEVICES=0  python cherry_seletion/data_by_cluster.py \
    --pt_data_path analysis/qwen2-multi-turn/wildchat_analysis.pt \
    --json_data_path data/wildchat-1M-en.json \
    --json_save_path analysis/qwen2-multi-turn/wildchat_analysis.json \
    --sample_num 10 \
    --kmeans_num_clusters 1000 \
    --low_th 25 \
    --up_th 75


