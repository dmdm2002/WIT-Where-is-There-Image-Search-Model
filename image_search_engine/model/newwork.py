import torch
import torch.nn as nn

from image_search_engine.model.image_encoder import CNNImageEncoder, SwinImageEncoder
from image_search_engine.model.keyword_encoder import KeywordEncoder


class ImageSearchModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, num_keywords: int):
        super().__init__()

        emb_size = self.get_emb_size(model_name=model_name)

        if model_name.lower() == 'swintransformer':
            self.image_encoder = SwinImageEncoder()
        else:
            self.image_encoder = CNNImageEncoder(model_name=model_name)

        self.keyword_encoder = KeywordEncoder(num_keywords=num_keywords, emb_size=emb_size)

        # self.weight_i = nn.Parameter(torch.ones(1))  # 첫 번째 모달 가중치
        # self.weight_k = nn.Parameter(torch.ones(1))  # 두 번째 모달 가중치

        self.fc_module_1 = nn.Sequential(
            nn.Linear(in_features=(emb_size * 2), out_features=emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ReLU(inplace=True),
        )

        self.fc_module_2 = nn.Sequential(
            nn.Linear(in_features=emb_size, out_features=emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ReLU(inplace=True),
        )

        self.fc_3 = nn.Linear(in_features=emb_size, out_features=num_classes)

    def get_emb_size(self, model_name):
        if model_name.lower() == 'resnet18':
            emb_size = 512
        elif model_name.lower() == 'resnet34':
            emb_size = 512
        elif model_name.lower() == 'resnet50':
            emb_size = 2048
        elif model_name.lower() == 'resnet101':
            emb_size = 2048
        elif model_name.lower() == 'resnet152':
            emb_size = 2048
        elif model_name.lower() == 'swintransformer':
            emb_size = 1536
        else:
            emb_size = 2048

        return emb_size

    def forward(self, x_i, x_k):
        x_i_enc = self.image_encoder(x_i) * 0.95
        x_k_enc = self.keyword_encoder(x_k) * 0.05
        # x_i_enc = self.image_encoder(x_i) * self.weight_i
        # x_k_enc = self.keyword_encoder(x_k) * self.weight_k

        x_enc = torch.concat([x_i_enc, x_k_enc], dim=1)

        x_enc = self.fc_module_1(x_enc)
        x_enc = self.fc_module_2(x_enc)

        result = self.fc_3(x_enc)

        return result
