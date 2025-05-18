import torch
from torch.utils.data import random_split, Subset
import numpy as np

def min_max_scale(data: torch.Tensor):
    return (data - data.min(dim=-1, keepdim=True)[0]) / (data.max(dim=-1, keepdim=True)[0] - data.min(dim=-1, keepdim=True)[0])

def standardize(data: torch.Tensor):
    std, mean = torch.std_mean(data, dim=-1, keepdim=True)
    return (data - mean) / std

def sequential_split(data, ratio: list[float|int]):
    total_ratio = round(sum(ratio), 5)
    copy_ratio = list(ratio)
    if (total_ratio != 1) and (total_ratio != len(data)):
        raise ValueError(f"Split ratio {copy_ratio} is not allowed!")
    if total_ratio == 1:
        for idx, val in enumerate(copy_ratio):
            copy_ratio[idx] = int(val * len(data) + copy_ratio[idx - 1])
    else:
        for idx, val in enumerate(copy_ratio):
            copy_ratio[idx] = int(val)
    
    copy_ratio.insert(0, 0)
    return [Subset(data, range(copy_ratio[pos], copy_ratio[pos+1])) for pos in range(len(copy_ratio) - 1)]

def metric(confusion_matrix:np.ndarray, *, verbose = False): # 2x2 array
    accuracy = sum([confusion_matrix[idx, idx] for idx in range(len(confusion_matrix))]) / confusion_matrix.sum()
    precision = confusion_matrix[-1, -1] / confusion_matrix[1, :].sum()
    recall = confusion_matrix[-1, -1] / confusion_matrix[:, 1].sum()
    f1_score = 2 * (precision * recall) / (precision + recall)
    if verbose:
        print(f"{accuracy=:.3f}")
        print(f"{precision=:.3f}")
        print(f"{recall=:.3f}")
        print(f"{f1_score=:.3f}")

    return accuracy, precision, recall, f1_score

def get_file_name(__file__):
    return __file__.split("\\")[-1][:-3]