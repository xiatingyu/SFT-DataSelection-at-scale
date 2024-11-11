import argparse
import os
import json
import torch


def parse_args():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--score_path', type=str,
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--data_path', type=str, nargs='+',
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--raw_data_path', type=str, 
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--output_path', type=str,
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--max_samples', type=int,
                           default=0, help='The maximum number of samples')


    args = argparser.parse_args()

    return args



if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    score = torch.load(args.score_path, map_location=device)
    raw_score = torch.load(args.raw_data_path, map_location=device)


    data = []
    for i in range(10):
        with open(args.data_path[0].format(str(i)), 'r') as file:
            print(args.data_path[0].format(str(i)) )
            data += json.load(file)
    print(len(data))
    
    graph_all = []
    ids_all = []

    graph_pos = []
    ids_pos = []

    graph_neg = []
    ids_neg = []
    assert len(data) == score.size(0)
    for i in range(score.size(0)):
        from_id = data[i]['ex_id']
        to_id = data[i]['id']
        weight = score[i]
        from_weight = raw_score[from_id]
        to_weight = raw_score[to_id]
    
        graph_all.append([from_id, to_id, weight, from_weight, to_weight])
        ids_all.append(from_id)
        ids_all.append(to_id)

        if weight < to_weight:
            graph_pos.append([from_id, to_id, weight, from_weight, to_weight])
            ids_pos.append(from_id)
            ids_pos.append(to_id)
        else:
            graph_pos.append([99999999, to_id, 99999999, 99999999, to_weight])
            ids_pos.append(to_id)
        
        if weight > to_weight:    
            graph_neg.append([from_id, to_id, weight, from_weight, to_weight])
            ids_neg.append(from_id)
            ids_neg.append(to_id)
        else:
            graph_neg.append([99999999, to_id, 99999999, 99999999, to_weight])
            ids_neg.append(to_id)
    
    node_key = {}
    idx_key = {}
    ids_all = list(set(ids_all))
    for i in range(len(ids_all)):
        idx_key[i] = ids_all[i]
        node_key[ids_all[i]] = i
    
    graph_dict_path = os.path.join(args.output_path, 'graph_dict.json')
    print(len(graph_all), len(ids_all))
    with open(graph_dict_path, 'w') as f:
        json.dump(idx_key, f, indent=2)

    graph_path = os.path.join(args.output_path, 'graph.txt')
    with open(graph_path, 'w') as f:
        for tuples in graph_all:
            f.write(f"{node_key[tuples[0]]}\t{node_key[tuples[1]]}\t{tuples[2]}\t{tuples[3]}\t{tuples[4]}\n")




    # node_key = {}
    # idx_key = {}
    # ids_pos = list(set(ids_pos))
    # for i in range(len(ids_pos)):
    #     idx_key[i] = ids_pos[i]
    #     node_key[ids_pos[i]] = i
    
    # graph_dict_path = os.path.join(args.output_path, 'graph_dict_pos.json')
    # print(len(graph_pos), len(ids_pos))
    # with open(graph_dict_path, 'w') as f:
    #     json.dump(idx_key, f, indent=2)

    # graph_path = os.path.join(args.output_path, 'graph_pos.txt')
    # with open(graph_path, 'w') as f:
    #     for tuples in graph_pos:
    #         if tuples[0] == 99999999:
    #             f.write(f"{tuples[0]}\t{node_key[tuples[1]]}\t{tuples[2]}\t{tuples[3]}\t{tuples[4]}\n")
    #         else:
    #             f.write(f"{node_key[tuples[0]]}\t{node_key[tuples[1]]}\t{tuples[2]}\t{tuples[3]}\t{tuples[4]}\n")

    
    # node_key = {}
    # idx_key = {}
    # ids_neg = list(set(ids_neg))
    # for i in range(len(ids_neg)):
    #     idx_key[i] = ids_neg[i]
    #     node_key[ids_neg[i]] = i

    # graph_dict_path = os.path.join(args.output_path, 'graph_dict_neg.json')
    # print(len(graph_neg), len(ids_neg))
    # with open(graph_dict_path, 'w') as f:
    #     json.dump(idx_key, f, indent=2)

    # graph_path = os.path.join(args.output_path, 'graph_neg.txt')
    # with open(graph_path, 'w') as f:
    #     for tuples in graph_neg:
    #         if tuples[0] == 99999999:
    #             f.write(f"{tuples[0]}\t{node_key[tuples[1]]}\t{tuples[2]}\t{tuples[3]}\t{tuples[4]}\n")
    #         else:
    #             f.write(f"{node_key[tuples[0]]}\t{node_key[tuples[1]]}\t{tuples[2]}\t{tuples[3]}\t{tuples[4]}\n")
