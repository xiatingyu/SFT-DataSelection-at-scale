import multiprocessing
import os
import json
import zlib
import torch
import time
import argparse
from collections import defaultdict

def load_original_data_pool(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def get_compression_ratio(input_data):
    data_str = str(input_data).encode('utf-8')
    compressed_data = zlib.compress(data_str, level=9)
    compressed_ratio = len(data_str) / len(compressed_data)
    return compressed_ratio

def selec_data_from_corpus(
    anchor_data,
    processed_data_index,
    budget,
    selection_num,
    candidate_budget='all',
    pool=None,
    turn_print=True,
    data_pool=None,
    global_information_redundancy_state=None,
):
    data_list = [data_pool[_] for _ in processed_data_index]

    selected_data = []
    selected_index = []
    if not turn_print:
        start_time = time.time()
    while True:
        if turn_print:
            start_time = time.time()
        # select topk instance to compute compression ratio in a greedy fashion
        if candidate_budget == 'all':
            group_information_redundancy_state = pool.map(get_compression_ratio, [anchor_data + selected_data + [part] for part in data_list])
            group_information_redundancy_state = torch.tensor(group_information_redundancy_state)
            group_information_redundancy_state[selected_index] = 1000000
            _, min_index = torch.topk(group_information_redundancy_state, k=selection_num, largest=False)
            new_index = min_index.tolist()
            selected_instance_list = []
            for _ in new_index:
                selected_instance = data_list[_]
                selected_instance_list.append(selected_instance)
            selected_index.extend(new_index)
            selected_data.extend(selected_instance_list)
        else:
            # global view
            _, cur_index = torch.topk(global_information_redundancy_state, k=candidate_budget, largest=False)
            group_list = [data_pool[idx] for idx in cur_index]
            # compute compression ratio
            group_information_redundancy_state = pool.map(get_compression_ratio, [anchor_data + selected_data + [part] for part in group_list])
            group_information_redundancy_state = torch.tensor(group_information_redundancy_state)
            global_information_redundancy_state[cur_index] = group_information_redundancy_state
            _, min_index = torch.topk(group_information_redundancy_state, k=selection_num, largest=False)
            new_index = cur_index[min_index].tolist()
            global_information_redundancy_state[new_index] = 1000000
            selected_instance_list = []
            for _ in new_index:
                selected_instance = data_pool[_]
                selected_instance_list.append(selected_instance)
            selected_index.extend(new_index)
            selected_data.extend(selected_instance_list)
        if turn_print:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Code execution time: {execution_time} seconds")

        cur_len = len(selected_data)
        if cur_len >= budget:
            if candidate_budget == 'all':
                selected_global_index = [processed_data_index[_] for _ in selected_index]
            else:
                selected_global_index = selected_index
            if not turn_print:
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Code execution time: {execution_time} seconds")
            return selected_global_index, selected_data

def ZIP_select(data_pool, save_path, sub_path, selected_data_path, budget, k1=10000, k2=200, k3=100, n_jobs=1):
    pool = multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), n_jobs))
    global_information_redundancy_state = pool.map(get_compression_ratio, [[part] for part in data_pool])
    global_information_redundancy_state = torch.tensor(global_information_redundancy_state)

    final_selected_data = []
    data_index = list(range(len(data_pool)))
    if selected_data_path is not None:
        with open(selected_data_path, 'r') as f:
            selected_data = json.load(f)
        data_id = [data['id'] for data in selected_data]
        final_selected_data.extend(selected_data)
        print(len(final_selected_data))
        cur_data_index = [_ for _ in data_index if _ not in data_id]
        print(len(cur_data_index))
    else:
        cur_data_index = list(range(len(data_pool)))
        
    while len(final_selected_data) < budget:
        print('stage 1 & stage 2')
        second_stage_index, _ = selec_data_from_corpus(
            final_selected_data, cur_data_index, k2, k2, k1, pool, turn_print=True,
            data_pool=data_pool,
            global_information_redundancy_state=global_information_redundancy_state
        )
        print('stage 3')
        third_stage_index, third_stage_data = selec_data_from_corpus(
            [], second_stage_index, k3, 1, 'all', pool, turn_print=False,
            data_pool=data_pool,
            global_information_redundancy_state=global_information_redundancy_state
        )
        cur_data_index = [_ for _ in cur_data_index if _ not in third_stage_index]
        final_selected_data.extend(third_stage_data)
        source_list = defaultdict(int)
        for _ in final_selected_data:
            source_list[_['source']] += 1
        print(f'selected {len(final_selected_data)}, including {source_list}')
        if len(final_selected_data) % 10000 == 0:
            print(f'selected {len(final_selected_data)}')
            with open(sub_path, 'w+', encoding='utf-8') as f:
                json.dump(final_selected_data, f, indent=4, ensure_ascii=False)
    with open(save_path, 'w+', encoding='utf-8') as f:
        json.dump(final_selected_data, f, indent=4, ensure_ascii=False)
    pool.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./pool.json', help='The path to the original data pool with sharegpt format.')
    parser.add_argument('--save_path', type=str, default='./zip_selected_data.json', help='The path to the selected dataset.')
    parser.add_argument('--sub_path', type=str, default='./zip_selected_data.json', help='The path to the selected dataset.')
    parser.add_argument('--selected_data_path', type=str, default='wildchat_zip_5w.json', help='The path to the selected dataset.')
    parser.add_argument('--budget', type=int, default=1000, help='The number of selected instances')
    parser.add_argument('--k1', type=int, default=10000, help='')
    parser.add_argument('--k2', type=int, default=200, help='The gpu index, -1 for cpu')
    parser.add_argument('--k3', type=int, default=100, help='The gpu index, -1 for cpu')
    parser.add_argument('--n_jobs', type=int, default=64, help='The number of jobs to use for calculating compression ratios.')
    args = parser.parse_args()
    original_data_pool = load_original_data_pool(args.data_path)
    ZIP_select(original_data_pool, args.save_path, args.sub_path, args.selected_data_path, args.budget, args.k1, args.k2, args.k3, args.n_jobs)