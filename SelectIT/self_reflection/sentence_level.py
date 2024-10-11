import json
import argparse
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import random
import numpy as np
import os
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def construction_rps(input_file, rp_file, k, start, end, tokenizer):
    f_rp = open(rp_file, 'r')
    promote = f_rp.readlines()
    instruction = "Instruction: "
    response = "Response: "
    ress = '\nThe answer is: '
    f = open(input_file, 'r')
    f = json.load(f)
    rating_prompt_list = []
    for item in tqdm(f[start:end]):
        for idx in range(k):
            rating_prompt = promote[idx]
            for i in range(0, len(item['conversations']), 2):
                rating_prompt += '\n' + instruction + item['conversations'][i]['value']
                rating_prompt +=  '\n' + response + item['conversations'][i+1]['value'] 

            tokenized = tokenizer(rating_prompt, truncation=True, max_length=8192, return_tensors="pt").to(device)
            # rating_prompt = promote[idx] + '\n' + instruction + ins + '\n' + response + res + ress
            rating_prompt = tokenizer.decode(tokenized.input_ids[0], skip_special_tokens=True)
            rating_prompt_list.append(rating_prompt + ress)
    return rating_prompt_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help='model name in the hub or local path')
    parser.add_argument('--input_file', type=str, required=True, help='input file')
    parser.add_argument('--rating_prompt_file', type=str, required=True, help='input file')
    parser.add_argument('--output_file', type=str, required=True, help='output file')
    parser.add_argument('--k', type=int, required=True, help='parameter')
    parser.add_argument('--proportion', type=float, required=True, default=0.2, help='parameter')
    parser.add_argument('--alpha', type=float, required=True, default=0.2, help='parameter')
    parser.add_argument('--start', type=int, default=0, help='parameter')
    parser.add_argument('--end', type=int, default=999999999, help='parameter')   
    args = parser.parse_args()
    model_name_or_path = args.model_name_or_path
    input_file = args.input_file
    rp_file = args.rating_prompt_file
    output_file = args.output_file
    proportion = args.proportion
    alpha = args.alpha
    k = args.k

    to_use_fast = False
    if "bloom" in model_name_or_path:
        to_use_fast = True
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=to_use_fast)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    rps = construction_rps(input_file, rp_file, k, args.start, args.end, tokenizer)

    print('okk')
    print(f'Loading Mater Model weights from path: {model_name_or_path}')
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
    print(model.hf_device_map)

    
    pro = []
    for idx, p in enumerate(tqdm(rps)):
        tokenized = tokenizer(p, padding=True, return_tensors="pt").to(device)
        tokenized.input_ids = tokenized.input_ids
        tokenized.attention_mask = tokenized.attention_mask
        with torch.no_grad():
            try:
                outputs = model(**tokenized)
                predictions = outputs[0]
                logits = predictions[:, -1, :]
                softmax_logits = torch.softmax(logits.float(), dim=-1)
                for index in range(1):
                    tmp_res = [float(softmax_logits[index][16]), float(softmax_logits[index][17]),
                               float(softmax_logits[index][18]), float(softmax_logits[index][19]),
                               float(softmax_logits[index][20])]
                    pro.append(tmp_res)
            except Exception as ex:
                print(ex)
    pro_softmax = []
    for item in tqdm(pro):
        tmp_pro_softmax = item
        tmp0_pro_softmax = []
        tmp1_pro_softmax = []
        for idx, item in enumerate(tmp_pro_softmax):
            tmp0_pro_softmax.append(np.exp(tmp_pro_softmax[idx] / sum(tmp_pro_softmax)))
        for jdx, item in enumerate(tmp0_pro_softmax):
            tmp1_pro_softmax.append(tmp0_pro_softmax[jdx] / sum(tmp0_pro_softmax))
        pro_softmax.append(tmp1_pro_softmax)

    data_num = int(len(pro_softmax) / k)
    sentence_level_score = []
    for idx in tqdm(range(data_num)):
        token_level_score = []
        for id in range(idx * k, (idx + 1) * k):
            score_base = np.argmax(pro_softmax[id])
            tmp_sum = 0
            for tmp_idx in range(k):
                tmp_sum += pro_softmax[id][score_base] - pro_softmax[id][tmp_idx]
            tmp_sum = tmp_sum / (k - 1)
            token_score = (score_base + 1) * tmp_sum
            token_level_score.append(token_score)
        avg = np.average(token_level_score)
        std = np.std(token_level_score)
        sentence_level_score.append(avg / (1 + alpha * std))


    final_it_data = []
    f = open(input_file, 'r')
    f = json.load(f)[args.start:args.end]

    for idx, item in enumerate(tqdm(sentence_level_score)):
        f[idx]['id'] = idx + args.start
        f[idx]['sentence_score'] = item
        final_it_data.append(f[idx])
   
    f_o = open(output_file, 'w')
    json.dump(final_it_data, f_o, ensure_ascii=False, indent=4)
    f_o.close()
