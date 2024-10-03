import os
from torch.utils.tensorboard import SummaryWriter


def logging_current_txt(path: str, info: dict, epoch: int):
    f = open(path, 'a', encoding='utf-8')

    info_list = info.items()  # dict2list
    f.write(f'----------------------[{epoch}]----------------------\n')
    for _, (key, value) in enumerate(info_list):
        f.write(f'{key}: {value}\n')

    return print('Finish Epoch Logging!!')


def logging_current_tensorboard(summary: SummaryWriter, info: dict, epoch: int, train: bool):
    if train:
        top_category = 'Train'
    else:
        top_category = 'Test'

    info_list = info.items()
    for _, (key, value) in enumerate(info_list):
        summary.add_scalar(f'{top_category}/{key}', value, epoch)

    return summary


def logging_best_score(path, info):
    f = open(path, 'a', encoding='utf-8')

    info_list = info.items()  # dict2list
    for _, (key, value) in enumerate(info_list):
        f.write(f'{key}: {value}\n')

    return print('Finish Best Epoch Logging!!')