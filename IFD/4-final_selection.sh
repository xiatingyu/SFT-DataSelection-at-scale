python data_merge.py \
        --pt_data_path "cherry_data/llama3/split/OpenHermes_pre" \
        --pt_save_path "cherry_data/llama3/OpenHermes_after_pre.pt"


CUDA_VISIBLE_DEVICES=0 python cherry_seletion/data_by_IFD.py \
        --pt_data_path "cherry_data/llama3-multi-turn/OpenHermes_after_pre.pt" \
        --model_name_or_path "llama3-8B-ifd-pre" \
        --json_data_path "data/OpenHermes2.5.json" \
        --json_save_path "cherry_data/llama3-multi-turn/OpenHermes_ifd_llama3_1w.json" \
        --max_length 4096 \
        --sample_number 10000 \
        --prompt alpaca

