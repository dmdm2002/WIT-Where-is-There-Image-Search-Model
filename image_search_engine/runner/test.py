import torch
import tqdm

from image_search_engine.model.newwork import ImageSearchModel
from image_search_engine.utils.dataset import get_loader
from image_search_engine.utils.functions import get_configs
from torchmetrics.classification import Accuracy


class Test:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = 'cpu'
        torch.manual_seed(self.cfg['seed'])
        torch.cuda.manual_seed_all(self.cfg['seed'])
        ckp_path = 'M:/Side/2024_TourAPI/backup/Swin_Keyword/try_3/ckp/36.pth'
        ckp = torch.load(ckp_path, map_location='cpu')

        self.model = ImageSearchModel(model_name=self.cfg['model_name'], num_classes=self.cfg['cls_num'], num_keywords=self.cfg['num_keywords']).to(self.device)
        self.model.load_state_dict(ckp['model_state_dict'])
        # backend = 'qnnpack'
        # model.qconfig = torch.quantization.get_default_qconfig(backend)
        # torch.backends.quantized.engine = backend
        # self.model_static_quantized = torch.quantization.prepare(model, inplace=False)
        # self.model_static_quantized = torch.quantization.convert(self.model_static_quantized, inplace=False)

        self.te_loader = get_loader(train=False,
                                    image_size=self.cfg['image_size'],
                                    batch_size=1,
                                    dataset_path=self.cfg['te_dataset_path'])

        self.acc_top1_metric = Accuracy(task='multiclass', num_classes=123, top_k=1).to(self.device)
        self.acc_top5_metric = Accuracy(task='multiclass', num_classes=123, top_k=5).to(self.device)
        self.acc_top10_metric = Accuracy(task='multiclass', num_classes=123, top_k=10).to(self.device)

    def run(self):
        te_scores = {'top1_acc': 0, 'top5_acc': 0, 'top10_acc': 0}

        with torch.no_grad():
            self.model.eval()
            for _, (image, keyword, label) in enumerate(
                    tqdm.tqdm(self.te_loader, desc=f"[Test]")):
                image = image.to(self.device)
                keyword = keyword.to(self.device)
                label = label.to(self.device)

                logits = self.model(image, keyword)

                self.acc_top1_metric.update(logits, label)
                self.acc_top5_metric.update(logits, label)
                self.acc_top10_metric.update(logits, label)

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
