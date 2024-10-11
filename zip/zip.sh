python ZIP.py \
    --data_path data/openhermes.json \
    --save_path openhermes_zip_10w.json \
    --budget 100000 \
    --selected_data_path openhermes_zip_5w.json \
    --sub_path openhermes_zip_x.json

# selected_data_path means the data that has been selected before, if you want to continue selecting data, you can set it to None
# sub_path means the data that has been selected during the process, save each 10000 data once.