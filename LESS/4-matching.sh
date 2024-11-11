DIM=8192
train_file_names=path/to/qwen2/Qwen2-7B-ckpt-{}
gradient_path=path/to/qwen2/Qwen2-7B-ckpt-{}/0/dim8192
ckpts="1 2 3 4"
checkpoint_weights="1.6854e-05 1.4868e-05 1.2483e-05 9.9999e-06"

validation_gradient_path=path/to/val/Qwen2-7B-ckpt{}/dim8192

output_path="path/to/selected_qwen2"

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

CUDA_VISIBLE_DEVICES=0  python3 -m less.data_selection.matching \
    --gradient_path $gradient_path \
    --train_file_names $train_file_names \
    --ckpts $ckpts \
    --checkpoint_weights $checkpoint_weights \
    --validation_gradient_path $validation_gradient_path \
    --output_path $output_path

