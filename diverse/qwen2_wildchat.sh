source activate dpo
pip install wandb==0.17.5
pip install sentence_transformers

export TOKENIZERS_PARALLELISM=false

DISTRIBUTED_ARGS="--nproc_per_node ${KUBERNETES_CONTAINER_RESOURCE_GPU} \
                    --nnodes ${WORLD_SIZE} \
                    --node_rank ${RANK} \
                    --master_addr ${MASTER_ADDR} \
                    --master_port 6655"


torchrun $DISTRIBUTED_ARGS train.py \
        --config_file ./configs/config_wildchat_qwen2.yml \
        --wandb_key cac878fde2ea0334527a15036103638df587d233 