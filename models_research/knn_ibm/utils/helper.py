import torch
from torch.utils.data import Subset
import numpy as np
import os

def min_max_scale(data: torch.Tensor) -> torch.Tensor:
    """
    Normalize the data to the range [0, 1] using min-max scaling.
    Assumes data is a 1D or 2D tensor (e.g., sequence of closing prices).
    """
    data_min = data.min(dim=-1, keepdim=True)[0]
    data_max = data.max(dim=-1, keepdim=True)[0]
    denominator = data_max - data_min
    # Avoid division by zero
    denominator = torch.where(denominator == 0, torch.tensor(1.0, device=denominator.device), denominator)
    return (data - data_min) / denominator

def standardize(data: torch.Tensor) -> torch.Tensor:
    """
    Standardize the data to have mean 0 and standard deviation 1.
    Assumes data is a 1D or 2D tensor (e.g., sequence of closing prices).
    """
    std, mean = torch.std_mean(data, dim=-1, keepdim=True)
    # Avoid division by zero
    std = torch.where(std == 0, torch.tensor(1.0, device=std.device), std)
    return (data - mean) / std

def sequential_split(dataset, ratio: list[float|int]):
    """
    Split the dataset sequentially into subsets based on the given ratio.
    The dataset is expected to be an instance of UpDownStockDataset.
    """
    total_ratio = round(sum(ratio), 5)
    copy_ratio = list(ratio)
    
    # Validate the ratio
    if (total_ratio != 1) and (total_ratio != len(dataset)):
        raise ValueError(f"Split ratio {copy_ratio} is not allowed! Sum should be 1 or match dataset length.")
    
    # Convert ratio to indices
    if total_ratio == 1:
        cumulative = 0
        for idx in range(len(copy_ratio)):
            cumulative += copy_ratio[idx] * len(dataset)
            copy_ratio[idx] = int(cumulative)
    else:
        for idx in range(len(copy_ratio)):
            copy_ratio[idx] = int(copy_ratio[idx])
    
    # Insert starting index
    copy_ratio.insert(0, 0)
    
    # Create subsets
    return [Subset(dataset, range(copy_ratio[pos], copy_ratio[pos + 1])) for pos in range(len(copy_ratio) - 1)]

def metric(confusion_matrix: np.ndarray, *, verbose: bool = False) -> tuple:
    """
    Compute classification metrics from a 2x2 confusion matrix.
    Confusion matrix format: [[TN, FP], [FN, TP]], where 1 is positive (price increases).
    """
    if confusion_matrix.shape != (2, 2):
        raise ValueError("Confusion matrix must be a 2x2 array.")
    
    # Calculate metrics
    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    precision = confusion_matrix[1, 1] / np.sum(confusion_matrix[1, :]) if np.sum(confusion_matrix[1, :]) != 0 else 0
    recall = confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1]) if np.sum(confusion_matrix[:, 1]) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    # Print metrics if verbose
    if verbose:
        print(f"accuracy={accuracy:.3f}")
        print(f"precision={precision:.3f}")
        print(f"recall={recall:.3f}")
        print(f"f1_score={f1_score:.3f}")
    
    return accuracy, precision, recall, f1_score

def get_file_name(file_path: str) -> str:
    """
    Extract the file name (without extension) from a file path.
    Works across different operating systems.
    """
    return os.path.splitext(os.path.basename(file_path))[0]