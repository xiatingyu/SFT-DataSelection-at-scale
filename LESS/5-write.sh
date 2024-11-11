

score_path="path/to/influence_score.pt"
data_path="path/to/openhermes.json"
output_path="path/to/openhermes_less_llama3_1w.json"
max_samples=10000


python3 -m less.data_selection.selected_data \
    --score_path $score_path \
    --data_path $data_path \
    --output_path $output_path \
    --max_samples $max_samples 

