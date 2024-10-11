import os
import json
import torch
import argparse
from tqdm import tqdm
import jsonlines 
import numpy as np


pt_data_list = []
for i in range(8):
    path = f'./data/qwen_data_{i}.pt'
    pt_data = torch.load(path, map_location=torch.device('cpu'))
    pt_data_list.extend(pt_data)

print(len(pt_data_list))

torch.save(pt_data_list, './data/qwen_data_merge.pt')
