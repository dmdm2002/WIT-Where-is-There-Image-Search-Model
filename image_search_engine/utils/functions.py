import yaml
import os
import torch


def get_configs(path):
    assert os.path.exists(path), f"경로[{path}]에 해당 파일이 존재하지 않습니다."
    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def save_configs(cfg):
    with open(f"{cfg['log_path']}/train_parameters.yml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def top_k_accuracy(output, target, k=5):
    """
    Computes the top-k accuracy for the specified values of k.

    Parameters:
    output (torch.Tensor): Model output, shape (batch_size, num_classes)
    target (torch.Tensor): Ground truth labels, shape (batch_size)
    k (int): The number of top elements to consider for accuracy

    Returns:
    float: Top-k accuracy
    """
    # Get the top k indices for each sample
    top_k = torch.topk(output, k, dim=1).indices

    # Expand target tensor to match top_k shape
    target_expanded = target.view(-1, 1).expand_as(top_k)

    # Check if the target is in the top k predictions
    correct = (top_k == target_expanded).sum().item()

    # Calculate accuracy
    accuracy = correct / target.size(0)

    return accuracy
