import json
import numpy as np
import argparse
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModel
from tqdm import tqdm
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

PROMPT_RAW = {
        "system": (
            "{system}"
        ),
        "user": (
            "\nHuman: {query}\nAssistant: "    
        ),
        "ass": (
            "{response}"
        )
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/cpfs01/shared/Group-m6/xiatingyu.xty/model/8b-base")
    parser.add_argument('--instruction_path', type=str, default="/cpfs01/shared/Group-m6/xiatingyu.xty/data/OpenHermes2.5.json")
    parser.add_argument('--save_path', type=str, default="save/openhermes/embeddings")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=10)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print('Load data....')
    data = []
    with open(args.instruction_path, "r") as f:
        # for line in f.readlines():
        #     data.append(json.loads(line))
        data = json.load(f) 
    
    data = data[args.start:args.end]

    ex_prompted = []
    for i in tqdm(range(len(data))):    
        encoded_messages=''
        con_len=len(data[i]['conversations'])
        for j in range(con_len-1):
            message = data[i]['conversations'][j]
            if message["from"] == "human":
                encoded_messages += PROMPT_RAW['user'].format(query=message['value'])
            else:
                encoded_messages += PROMPT_RAW['ass'].format(response=message['value'])
        
        encoded_messages = encoded_messages.strip()
        response = PROMPT_RAW['ass'].format(response=data[i]['conversations'][-1]['value'])
        ex_prompted.append((encoded_messages, response))
    
    query_num, response_num = [], []
    for query, response in tqdm(ex_prompted):
        tokens = tokenizer(query, truncation=True, return_tensors='pt').to(device)
        index = torch.count_nonzero(tokens['input_ids']).item()
        query_num.append(index)

        tokens = tokenizer(response, truncation=True, return_tensors='pt').to(device)
        index = torch.count_nonzero(tokens['input_ids']).item()
        response_num.append(index)

    assert len(query_num)==len(response_num)
    assert len(query_num)==len(data)
    for i in tqdm(range(len(data))):    
        data[i]['query_token'] = query_num[i]
        data[i]['response_token'] = response_num[i]
        data[i]['total_token'] = query_num[i]+response_num[i]

    with open(args.save_path, 'w') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

