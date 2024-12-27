import random
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
import math

def init_seeds(seed=0, cuda_deterministic=True):
    """
    Initialize the random number seed
    :param seed: random number seed
    :param cuda_deterministic: Whether to fix the random number seed of cuda
    Setting this flag to True allows us to pre-optimize the convolutional layers of the model in PyTorch
    We can't set cudnn.benchmark=True if our network model keeps changing. Because it takes time to find the optimal convolution algorithm.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

def xavier_init(model):
    """
    Initialize the model parameters
    :param model: the model
    """
    for name, par in model.named_parameters():
        if 'weight' in name and len(par.shape) >= 2:
            nn.init.xavier_normal_(par)
        elif 'bias' in name:
            nn.init.constant_(par, 0.0)

def evaluate_function(output, positives, negatives, neg_num):
    result = [{} for _ in range(len(output))]
    for i in range(len(output)):
        neg_num_i = int(neg_num[i])
        all_negatives = negatives[i][:neg_num_i]
        pos_score = output[i][positives[i]]
        neg_scores = output[i][all_negatives]
        success = ((neg_scores - pos_score) < 0).sum()
        success = int(success)
        if neg_num_i - success < 5:
            result[i]['recall@5'] = 1.0
            result[i]['mrr@5'] = 1 / (neg_num_i - success + 1)
            result[i]['ndcg@5'] = 1 / (math.log((neg_num_i - success + 2), 2))
        else:
            result[i]['recall@5'] = 0.0
            result[i]['mrr@5'] = 0.0
            result[i]['ndcg@5'] = 0.0
        if neg_num_i - success < 10:
            result[i]['recall@10'] = 1.0
            result[i]['mrr@10'] = 1 / (neg_num_i - success + 1)
            result[i]['ndcg@10'] = 1 / (math.log((neg_num_i - success + 2), 2))
        else:
            result[i]['recall@10'] = 0.0
            result[i]['mrr@10'] = 0.0
            result[i]['ndcg@10'] = 0.0
        if neg_num_i - success < 20:
            result[i]['recall@20'] = 1.0
            result[i]['mrr@20'] = 1 / (neg_num_i - success + 1)
            result[i]['ndcg@20'] = 1 / (math.log((neg_num_i - success + 2), 2))
        else:
            result[i]['recall@20'] = 0.0
            result[i]['mrr@20'] = 0.0
            result[i]['ndcg@20'] = 0.0
    return result

def get_metrics(metrics_name,total_result):
    if metrics_name == 'recall@5':
        recall5 = 0.0
        for i in total_result:
            recall5 += i['recall@5']
        return recall5 / len(total_result)
    elif metrics_name == 'mrr@5':
        mrr5 = 0.0
        for i in total_result:
            mrr5 += i['mrr@5']
        return mrr5 / len(total_result)
    elif metrics_name == 'ndcg@5':
        ndcg5 = 0.0
        for i in total_result:
            ndcg5 += i['ndcg@5']
        return ndcg5 / len(total_result)

    elif metrics_name == 'recall@10':
        recall10 = 0.0
        for i in total_result:
            recall10 += i['recall@10']
        return recall10 / len(total_result)
    elif metrics_name == 'mrr@10':
        mrr10 = 0.0
        for i in total_result:
            mrr10 += i['mrr@10']
        return mrr10 / len(total_result)
    elif metrics_name == 'ndcg@10':
        ndcg10 = 0.0
        for i in total_result:
            ndcg10 += i['ndcg@10']
        return ndcg10 / len(total_result)

    elif metrics_name == 'recall@20':
        recall20 = 0.0
        for i in total_result:
            recall20 += i['recall@20']
        return recall20 / len(total_result)
    elif metrics_name == 'mrr@20':
        mrr20 = 0.0
        for i in total_result:
            mrr20 += i['mrr@20']
        return mrr20 / len(total_result)
    elif metrics_name == 'ndcg@20':
        ndcg20 = 0.0
        for i in total_result:
            ndcg20 += i['ndcg@20']
        return ndcg20 / len(total_result)
    else:
        raise Exception("error!")

def stat_operation(full_layer_output_ori, leave_one_seq):
    # Calculate the proportion of keep operation, delete operation and insert operation
    operation = []
    for i in range(len(full_layer_output_ori)):
        decision = full_layer_output_ori[i][leave_one_seq[i] != 0]
        operation.append([int((decision == 0).sum()), int((decision == 1).sum()), int((decision == 2).sum())])
    return operation
    
def seqs_normalization(modified_seqs, modified_max_seqs_len, item_num):

    modified_padding_seqs = []
    for k, v in enumerate(modified_seqs):
        v = list(set(v))
        v1 = [num for num in v if num != item_num-1 and num != item_num-2 and num != 0]
        
        if len(v1) > modified_max_seqs_len:
            padding_seqs = v1[-modified_max_seqs_len:]
            modified_padding_seqs.append(padding_seqs)
        elif 0 < len(v1) <= modified_max_seqs_len:
            padding_seqs = v1.copy()
            padding_seqs += [0] * (modified_max_seqs_len-len(padding_seqs))
            modified_padding_seqs.append(padding_seqs)
        else:
            padding_seqs = v1 + [item_num-2] + [0] * (modified_max_seqs_len-1)
            modified_padding_seqs.append(padding_seqs)

    modified_padding_seqs = torch.tensor(modified_padding_seqs, dtype=torch.long)
    return modified_padding_seqs
