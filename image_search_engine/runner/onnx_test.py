import numpy as np
import torch
import tqdm
import onnx
import onnxruntime

from image_search_engine.utils.dataset import get_loader
from torchmetrics.classification import Accuracy
from image_search_engine.utils.functions import get_configs


class Test:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        torch.manual_seed(self.cfg['seed'])
        torch.cuda.manual_seed_all(self.cfg['seed'])

        model = onnx.load('../../model_weight/ImageSearchModel_static_quantized_2.onnx')
        print(onnx.checker.check_model(model))

        self.ort_session = onnxruntime.InferenceSession('../../model_weight/ImageSearchModel_static_quantized_2.onnx',
                                                        providers=['CPUExecutionProvider'])

        self.te_loader = get_loader(train=False,
                                    image_size=self.cfg['image_size'],
                                    batch_size=1,
                                    dataset_path=self.cfg['te_dataset_path'])

        self.acc_top1_metric = Accuracy(task='multiclass', num_classes=123, top_k=1)
        self.acc_top5_metric = Accuracy(task='multiclass', num_classes=123, top_k=5)
        self.acc_top10_metric = Accuracy(task='multiclass', num_classes=123, top_k=10)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def run(self):
        te_scores = {'top1_acc': 0, 'top5_acc': 0, 'top10_acc': 0}
        for _, (image, keyword, label) in enumerate(
                tqdm.tqdm(self.te_loader, desc=f"[Test]")):
            ort_inputs = {self.ort_session.get_inputs()[0].name: self.to_numpy(image),
                          self.ort_session.get_inputs()[1].name: self.to_numpy(keyword)}

            output_name = [output.name for output in self.ort_session.get_outputs()]

            logits = self.ort_session.run(output_name, ort_inputs)

            self.acc_top1_metric.update(torch.tensor(logits[0]), label)
            self.acc_top5_metric.update(torch.tensor(logits[0]), label)
            self.acc_top10_metric.update(torch.tensor(logits[0]), label)

        te_scores['top1_acc'] = self.acc_top1_metric.compute().item()
        te_scores['top5_acc'] = self.acc_top5_metric.compute().item()
        te_scores['top10_acc'] = self.acc_top10_metric.compute().item()
        self.acc_top1_metric.reset()
        self.acc_top5_metric.reset()
        self.acc_top10_metric.reset()

        print('---------------------------------------------------------------------')
        print(f"||[Test] Top-1 Acc: {te_scores['top1_acc']} | Top-5 Acc: {te_scores['top5_acc']} | Top-10 Acc: {te_scores['top10_acc']}||")
        print('---------------------------------------------------------------------\n')


if __name__ == '__main__':
    cfg = get_configs('../configs/train_configs.yml')
    test = Test(cfg)

    test.run()
