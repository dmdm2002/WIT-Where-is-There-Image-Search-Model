from image_search_engine.utils.functions import get_configs
from image_search_engine.runner.train import Train

import gc
import torch


if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()

    cfg = get_configs('./image_search_engine/configs/train_configs.yml')
    train = Train(cfg)

    train.run()
