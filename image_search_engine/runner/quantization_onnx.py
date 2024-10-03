import torch
import os
import torch.nn as nn

import onnxruntime
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantType, QuantFormat

from image_search_engine.model.newwork import ImageSearchModel
from image_search_engine.utils.dataset import get_loader


def convert_onnx(out_path):
    dummy_image_input = torch.randn(1, 3, 224, 224, device='cpu')
    dummy_keyword_input = torch.randn(1, 17, device='cpu')
    model = ImageSearchModel('swintransformer', 123, 17)
    ckp_path = 'M:/Side/2024_TourAPI/backup/Swin_Keyword/try_3/ckp/36.pth'
    ckp = torch.load(ckp_path, map_location='cpu')
    model.load_state_dict(ckp['model_state_dict'])
    model.eval()

    torch.onnx.export(model,
                      (dummy_image_input, dummy_keyword_input),
                      out_path,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=14,  # the ONNX version to export the model to
                      do_constant_folding=True,  # constant folding for optimization
                      input_names=['image_input', 'keyword_input'],
                      output_names=['output'],
                      )

    return print('Convert ONNX Finsh!! [.pth --> .onnx]')


def quantization_dynamic(fp32_onnx_path, quantized_model_path):
    quantize_dynamic(fp32_onnx_path, quantized_model_path, weight_type=QuantType.QUInt8)

    return print("Dynamic Quantization Finsh!! [fp32 --> fp8]")


def quantization_static(fp32_onnx_path, quantized_model_path):
    # CustomDataset 인스턴스 생성
    te_loader = get_loader(train=False,
                           image_size=224,
                           batch_size=1,
                           dataset_path="M:/Side/2024_TourAPI/DB/test.csv")
    dr = MyCalibrationDataReader(te_loader)

    q_static_opts = {"ActivationSymmetric": False,
                     "WeightSymmetric": False}

    quantize_static(fp32_onnx_path,
                    quantized_model_path,
                    calibration_data_reader=dr,
                    weight_type=QuantType.QUInt8,
                    activation_type=QuantType.QUInt8,
                    quant_format=QuantFormat.QDQ,
                    per_channel=True,  # Per-channel 양자화를 비활성화
                    reduce_range=True,  # 양자화 범위 축소를 비활성화
                    extra_options=q_static_opts
                    )

    return print("Static Quantization Finsh!! [fp32 --> fp8]")


class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.enum_data = None

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def get_next(self):
        if not self.enum_data:
            self.enum_data = iter(self.dataloader)
        try:
            image, keyword, label = next(self.enum_data)
            # ONNX 모델에 맞게 입력 데이터를 변환합니다.
            # 예시에서는 'input_image'와 'input_keyword'라는 입력 이름을 가정합니다.
            return {'image_input': self.to_numpy(image), 'keyword_input': self.to_numpy(keyword)}
        except StopIteration:
            return None


if __name__ == '__main__':
    fp32_onnx_path = '../../model_weight/ImageSearchModel.onnx'
    # convert_onnx(fp32_onnx_path)

    quantized_model_path = '../../model_weight/ImageSearchModel_static_quantized_2.onnx'
    quantization_static(fp32_onnx_path, quantized_model_path)
