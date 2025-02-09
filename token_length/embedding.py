import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModel
import json
from tqdm import tqdm
import numpy as np
import os
import argparse

def embed_texts_batched(texts, batch_size=10):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", truncation=True, padding='max_length', max_length=4096)
        tokens = {k: v.cuda() for k, v in tokens.items()}
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        all_embeddings.extend(embeddings)
    return all_embeddings



def load_sample(file_name):
    if file_name.endswith('.json'):
        with open(file_name, "r") as f:
            data = json.load(f)
    elif file_name.endswith('.jsonl'):
        data = []
        with open(file_name, "r") as f:
            for line in f.readlines():
                data.append(json.loads(line))
    
    ex_prompted = []
    for item in data:
        conv = item['conversations']
        line = ''
        for tmp in conv:
            if tmp['from'] == 'human':
                line += '\nHuman: ' + tmp['value'] 
            elif tmp['from'] == 'gpt':
                line += '\nAssistant: ' + tmp['value']
        
        ex_prompted.append(line.strip())
    return ex_prompted


# 初始化模型和分词器 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/root/autodl-tmp/model/Qwen2-7B")
    parser.add_argument('--instruction_path', type=str, default="/root/autodl-tmp/fbnm-journal/data/alpaca_gpt4.json")
    parser.add_argument('--save_embedding_path', type=str, default="embeddings.npy")
    args = parser.parse_args()

    MODEL_PATH = args.model_path 
    INSTRUCTION_PATH = args.instruction_path
    SAVE_EMBEDDING_PATH = args.save_embedding_path

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            # load_in_8bit=in_8bit,
        )
    model.eval()


    # load sample
    sample = load_sample(INSTRUCTION_PATH)
    print(sample[0])
    print("START EMBEDDING ..."*3)
    embeddings = embed_texts_batched(sample)
    print(len(embeddings))
    np.save(f'{SAVE_EMBEDDING_PATH}/{len(sample)}.npy', embeddings)