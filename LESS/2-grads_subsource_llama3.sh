# source activate sft

# path=/cpfs01/data/shared/Group-m6/xiatingyu.xty/data/jsonl
# for file in $path/*; do
#     tmp=${file#*OpenHermes2.5_}
#     file_name=${tmp%*.jsonl}
#     for ((k = 1; k < 5; k++)); do
#         bash llama3.sh $file_name $k
#     done
# done
for ((k = 1; k < 5; k++)); do
    bash llama3.sh  $k
done

