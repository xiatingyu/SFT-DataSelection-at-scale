import argparse
import os
import json
import torch
import copy

def parse_args():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--score_path', type=str,
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--data_path', type=str,
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--output_path', type=str,
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--max_samples', type=int,
                           default=None, help='The maximum number of samples')


    args = argparser.parse_args()

    return args



if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    score = torch.load(args.score_path, map_location=device)
    # length = score.size(0)
    # if args.max_samples > length:
    #     args.max_samples = length
    # vals, indices = score.topk(k=args.max_samples, largest=True, sorted=True)


    data = []
    print(args.data_path)
    with open(args.data_path, 'r') as file:
        data = json.load(file)
        # for line in file:
        #     data.append(json.loads(line))
    assert len(data) == score.size(0)
    final = []
    for i in range(len(data)):
        tmp = copy.deepcopy(data[i])
        tmp['score'] = score[i].item()
        final.append(tmp)
    
    with open(args.output_path, 'w') as file:
        json.dump(final, file, ensure_ascii=False, indent=4)

    