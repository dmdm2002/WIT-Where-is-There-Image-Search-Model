import timm
import torch.nn as nn
import torchvision.models as models


class CNNImageEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        if model_name.lower() == 'resnet18':
            encoder = models.resnet18(pretrained=True)
        elif model_name.lower() == 'resnet34':
            encoder = models.resnet34(pretrained=True)
        elif model_name.lower() == 'resnet50':
            encoder = models.resnet50(pretrained=True)
        elif model_name.lower() == 'resnet101':
            encoder = models.resnet101(pretrained=True)
        elif model_name.lower() == 'resnet152':
            encoder = models.resnet152(pretrained=True)
        else:
            print(
                "Error encoder model!!!, You can select a model in  this list ==> [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152]")
            print("Select default Model: [ResNet50]")
            encoder = models.resnet50(pretrained=True)

        self.stem = nn.Sequential(encoder.conv1,
                                  encoder.bn1,
                                  encoder.relu,
                                  encoder.maxpool)

        self.res_block_1 = encoder.layer1
        self.res_block_2 = encoder.layer2
        self.res_block_3 = encoder.layer3
        self.res_block_4 = encoder.layer4

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        stem = self.stem(x)

        res_out_1 = self.res_block_1(stem)
        res_out_2 = self.res_block_2(res_out_1)
        res_out_3 = self.res_block_3(res_out_2)
        res_out_4 = self.res_block_4(res_out_3)

        out = self.pool(res_out_4)
        out = out.view(out.size(0), -1)

        return out


class SwinImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('swin_large_patch4_window7_224.ms_in22k', num_classes=0, pretrained=True)
        # self.linear = nn.Linear(1536, 1024)

    def forward(self, x):
        # x = self.model(x)
        return self.model(x)


class ViTImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('vit_base_patch8_224', num_classes=0, pretrained=True)
        # self.linear = nn.Linear(1536, 1024)

    def forward(self, x):
        # x = self.model(x)
        return self.model(x)


if __name__ == '__main__':
    a = SwinImageEncoder()

