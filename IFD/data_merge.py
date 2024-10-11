import json
import heapq
import pandas as pd
import numpy as np
import random
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_data_path", type=str, default='/cpfs01/shared/Group-m6/xiatingyu.xty/Cherry_LLM/openhermes/llama3/OpenHermes_pre_anlysis')
    parser.add_argument("--pt_save_path", type=str, default='/cpfs01/shared/Group-m6/xiatingyu.xty/Cherry_LLM/openhermes/llama3/OpenHermes_after_pre.pt')
    args = parser.parse_args()
    return args
   
if __name__ == '__main__':
    args = parse_args()
    print(args)

    all_data = []
    for i in range(8):
        load_path = args.pt_data_path + "_{}.pt".format(i)
        pt_data = torch.load(load_path, map_location=torch.device('cpu'))
        all_data.extend(pt_data)
        print(len(pt_data))
    
    print(len(all_data))
    torch.save(all_data, args.pt_save_path)

    

   