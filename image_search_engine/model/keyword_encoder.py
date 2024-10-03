import torch.nn as nn


class KeywordEncoder(nn.Module):
    def __init__(self, num_keywords, emb_size):
        super().__init__()

        self.fc_1 = nn.Linear(num_keywords, 1024)
        self.bn_1 = nn.BatchNorm1d(1024)
        self.relu_1 = nn.ReLU(inplace=True)

        self.fc_2 = nn.Linear(1024, emb_size)
        self.bn_2 = nn.BatchNorm1d(emb_size)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.fc_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        return x