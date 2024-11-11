import argparse
import os

import torch

argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training')
argparser.add_argument('--gradient_path', type=str, default="{} ckpt{}",
                       help='The path to the gradient file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                       help='The name of the training file')
argparser.add_argument('--ckpts', type=int, nargs='+',
                       help="Checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                       help="checkpoint weights")
argparser.add_argument('--target_task_names', type=str,
                       nargs='+', help="The name of the target tasks")
argparser.add_argument('--validation_gradient_path', type=str,
                       default="{} ckpt{}", help='The path to the validation gradient file')
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')


args = argparser.parse_args()

N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 1000}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score.

    Args:
        training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
        validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
    """
    # N x N_VALID
    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1))
    return influence_scores


# renormalize the checkpoint weights
if sum(args.checkpoint_weights) != 1:
    s = sum(args.checkpoint_weights)
    args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

# calculate the influence score for each validation task

influence_score_list = None
for j in range(10):
    influence_score = 0
    for i, ckpt in enumerate(args.ckpts):
        validation_path = args.validation_gradient_path.format(
            ckpt)
        if os.path.isdir(validation_path):
            validation_path = os.path.join(validation_path, "all_orig.pt")
        validation_info = torch.load(validation_path)
        if not torch.is_tensor(validation_info):
            validation_info = torch.tensor(validation_info)
        validation_info = validation_info.to(device).float()

        ckpt_dir = args.train_file_names[0].format(ckpt)
        training_info = None
        for sub_dir in range(j*8, (j+1)*8):
            gradient_path = os.path.join(os.path.join(ckpt_dir, str(sub_dir)), "dim8192")
            print(gradient_path)
            if os.path.isdir(gradient_path):
                gradient_path = os.path.join(gradient_path, "all_orig.pt")
            if training_info is None:
                training_info = torch.load(gradient_path)
            else:
                training_info = torch.cat(
                    (training_info, torch.load(gradient_path)), dim=0)
        
        print("ckpt", ckpt)
        print(training_info.shape)

        if not torch.is_tensor(training_info):
            training_info = torch.tensor(training_info)
        training_info = training_info.to(device).float()

        influence_score += args.checkpoint_weights[i] * \
            calculate_influence_score(
                training_info=training_info, validation_info=validation_info)
        print(influence_score.shape)

    # influence_score = influence_score.reshape(
    #     influence_score.shape[0], 1000, -1).mean(-1).max(-1)[0]
    influence_score = influence_score.mean(-1)
    
    if influence_score_list is None:
        influence_score_list = influence_score
    else:
        influence_score_list = torch.cat(  
            (influence_score_list, influence_score), dim=0)
    
    print(influence_score_list.shape)



if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
output_file = os.path.join(
    args.output_path, "influence_score_icl.pt")
torch.save(influence_score_list, output_file)
print("Saved influence score to {}".format(output_file))
