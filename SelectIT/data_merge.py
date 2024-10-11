import json
import heapq
import pandas as pd
import numpy as np
import random
import copy


def llama3():
    with open("data/wildchat-1M-en.json", 'r') as f:
        raw_data = json.load(f)
    all_data=[]
    for i in range(8):
        with open('wild/llama/split/sentence_{}.json'.format(i), 'r') as f:
            data = json.load(f)
            print(len(data))
            for item in data:
                assert item['conversations'][0]['value'] == raw_data[item['id']]['conversations'][0]['value']
                all_data.append(item)
            
                
    print(len(all_data))
    with open('wild/llama/sentence_score.json', 'w') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    print(len(all_data))

    data = []
    with open('wild/llama/sentence_score.json', 'r') as f:
        data = json.load(f)
    all_data = sorted(data, key=lambda x: x['sentence_score'], reverse=True)

    final = all_data[:10000]
    random.shuffle(final)
    with open('wild/llama/select_llama3_10000.json', 'w') as f: 
        json.dump(final, f, ensure_ascii=False, indent=4)
    
    final = all_data[:50000]
    random.shuffle(final)
    with open('wild/llama/select_llama3_50000.json', 'w') as f: 
        json.dump(final, f, ensure_ascii=False, indent=4)



    return


def qwen():
    with open("/cpfs01/data/shared/Group-m6/xiatingyu.xty/data/wildchat-1M-en.json", 'r') as f:
        raw_data = json.load(f)
    # model level
    sentence1 = []
    sentence2 = []
    sentence3 = []
    model_data = []

    for i in range(8):
        with open('wild/qwen2/1.5/split/sentence_{}.json'.format(i), 'r') as f:
            data = json.load(f)
            print(len(data))
            for item in data:
                assert item['conversations'][0]['value'] == raw_data[item['id']]['conversations'][0]['value']
                sentence1.append(item)
    print(len(sentence1))
    for i in range(8):
        with open('wild/qwen2/7/sentence_{}.json'.format(i), 'r') as f:
            data = json.load(f)
            print(len(data))
            for item in data:
                assert item['conversations'][0]['value'] == raw_data[item['id']]['conversations'][0]['value']
                sentence2.append(item)
    

    
    final = []
    for j in range(len(sentence1)):
        tmp = copy.deepcopy(sentence1[j])
        del tmp['sentence_score']
        score = sentence1[j]["sentence_score"]*(1.5/8.5) + sentence2[j]["sentence_score"]*(7/8.5) #+ sentence3[j]["sentence_score"]*(7/12.8)
        tmp['score'] = score
        final.append(tmp)
    
    
    with open('wild/qwen2/wildchat_select_qwen.json', 'w') as f:
        json.dump(final, f, ensure_ascii=False, indent=4)
    
    final = sorted(final, key=lambda x: x['score'], reverse=True)
    

    return



if __name__ == '__main__':
    llama3()
    qwen()
   