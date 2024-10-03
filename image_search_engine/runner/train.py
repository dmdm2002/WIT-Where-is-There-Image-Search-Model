import os
import torch
import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from image_search_engine.model.newwork import ImageSearchModel
from image_search_engine.utils.dataset import get_loader
from image_search_engine.utils.logger import logging_current_txt, logging_current_tensorboard, logging_best_score
from image_search_engine.utils.functions import save_configs
from torchmetrics.classification import Accuracy


class Train:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        torch.manual_seed(self.cfg['seed'])
        torch.cuda.manual_seed_all(self.cfg['seed'])

        if self.cfg['do_logging']:
            os.makedirs(f"{self.cfg['log_path']}/tensorboard", exist_ok=True)
            self.summary = SummaryWriter(f"{self.cfg['log_path']}/tensorboard")
            save_configs(self.cfg)
        if self.cfg['do_ckp_save']:
            os.makedirs(f"{self.cfg['ckp_path']}", exist_ok=True)

        self.model = ImageSearchModel(model_name=self.cfg['model_name'], num_classes=self.cfg['cls_num'], num_keywords=self.cfg['num_keywords']).to(self.cfg['device'])
        self.optimizer = optim.AdamW(self.model.parameters(), self.cfg['lr'], (self.cfg['b1'], self.cfg['b2']), weight_decay=self.cfg['weight_decay'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.cfg['epoch'], eta_min=0)

        self.tr_loader = get_loader(train=True,
                                    image_size=self.cfg['image_size'],
                                    crop=self.cfg['crop'],
                                    jitter=self.cfg['jitter'],
                                    noise=self.cfg['noise'],
                                    batch_size=self.cfg['batch_size'],
                                    dataset_path=self.cfg['tr_dataset_path'])

        self.te_loader = get_loader(train=False,
                                    image_size=self.cfg['image_size'],
                                    dataset_path=self.cfg['te_dataset_path'])

        self.criterion = nn.CrossEntropyLoss()
        self.acc_top1_metric = Accuracy(task='multiclass', num_classes=123, top_k=1).to(self.cfg['device'])
        self.acc_top5_metric = Accuracy(task='multiclass', num_classes=123, top_k=5).to(self.cfg['device'])
        self.acc_top10_metric = Accuracy(task='multiclass', num_classes=123, top_k=10).to(self.cfg['device'])

        self.best_top1_score = {'epoch': 0, 'top1_acc': 0, 'top5_acc': 0, 'top10_acc': 0}
        self.best_top5_score = {'epoch': 0, 'top1_acc': 0, 'top5_acc': 0, 'top10_acc': 0}
        self.best_top10_score = {'epoch': 0, 'top1_acc': 0, 'top5_acc': 0, 'top10_acc': 0}

    def run(self):
        tr_scores = {'top1_acc': 0, 'top5_acc': 0, 'top10_acc': 0, 'loss': 0}
        te_scores = {'top1_acc': 0, 'top5_acc': 0, 'top10_acc': 0, 'loss': 0}

        for ep in range(self.cfg['epoch']):
            self.model.train()
            for _, (image, keyword, label) in enumerate(
                    tqdm.tqdm(self.tr_loader, desc=f"[Train-->{ep}/{self.cfg['epoch']}]")):
                image = image.to(self.cfg['device'])
                keyword = keyword.to(self.cfg['device'])
                label = label.to(self.cfg['device'])

                logits = self.model(image, keyword)
                loss = self.criterion(logits, label)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.acc_top1_metric.update(logits, label)
                self.acc_top5_metric.update(logits, label)
                self.acc_top10_metric.update(logits, label)
                tr_scores['loss'] += loss.item()

            tr_scores['top1_acc'] = self.acc_top1_metric.compute().item()
            tr_scores['top5_acc'] = self.acc_top5_metric.compute().item()
            tr_scores['top10_acc'] = self.acc_top10_metric.compute().item()
            self.acc_top1_metric.reset()
            self.acc_top5_metric.reset()
            self.acc_top10_metric.reset()
            tr_scores['loss'] = tr_scores['loss'] / len(self.tr_loader)

            with torch.no_grad():
                self.model.eval()
                for _, (image, keyword, label) in enumerate(
                        tqdm.tqdm(self.te_loader, desc=f"[Train-->{ep}/{self.cfg['epoch']}]")):
                    image = image.to(self.cfg['device'])
                    keyword = keyword.to(self.cfg['device'])
                    label = label.to(self.cfg['device'])

                    logits = self.model(image, keyword)
                    loss = self.criterion(logits, label)

                    self.acc_top1_metric.update(logits, label)
                    self.acc_top5_metric.update(logits, label)
                    self.acc_top10_metric.update(logits, label)
                    te_scores['loss'] += loss.item()

                te_scores['top1_acc'] = self.acc_top1_metric.compute().item()
                te_scores['top5_acc'] = self.acc_top5_metric.compute().item()
                te_scores['top10_acc'] = self.acc_top10_metric.compute().item()
                self.acc_top1_metric.reset()
                self.acc_top5_metric.reset()
                self.acc_top10_metric.reset()
                te_scores['loss'] = tr_scores['loss'] / len(self.tr_loader)

            if self.best_top1_score['top1_acc'] <= te_scores['top1_acc']:
                self.best_top1_score['top1_acc'] = te_scores['top1_acc']
                self.best_top1_score['top5_acc'] = te_scores['top5_acc']
                self.best_top1_score['top10_acc'] = te_scores['top10_acc']
                self.best_top1_score['epoch'] = ep

            if self.best_top5_score['top5_acc'] <= te_scores['top5_acc']:
                self.best_top5_score['top1_acc'] = te_scores['top1_acc']
                self.best_top5_score['top5_acc'] = te_scores['top5_acc']
                self.best_top5_score['top10_acc'] = te_scores['top10_acc']
                self.best_top5_score['epoch'] = ep

            if self.best_top10_score['top10_acc'] <= te_scores['top10_acc']:
                self.best_top10_score['top1_acc'] = te_scores['top1_acc']
                self.best_top10_score['top5_acc'] = te_scores['top5_acc']
                self.best_top10_score['top10_acc'] = te_scores['top10_acc']
                self.best_top10_score['epoch'] = ep

            if self.cfg['do_print']:
                print('---------------------------------------------------------------------')
                print(f"[Epoch: {ep}/{self.cfg['epoch']}]")
                print(f"||[Train] Top-1 Acc: {tr_scores['top1_acc']} | Top-5 Acc: {tr_scores['top5_acc']} | Top-10 Acc: {tr_scores['top10_acc']}||")
                print(f"||[Test] Top-1 Acc: {te_scores['top1_acc']} | Top-5 Acc: {te_scores['top5_acc']} | Top-10 Acc: {te_scores['top10_acc']}||")
                print(f"||Top-1 Acc Best epoch {self.best_top1_score['epoch']}|| --> Best Acc: {self.best_top1_score['top1_acc']}")
                print(f"||Top-5 Acc Best epoch {self.best_top5_score['epoch']}|| --> Best Acc: {self.best_top5_score['top5_acc']}")
                print(f"||Top-10 Acc Best epoch {self.best_top10_score['epoch']}|| --> Best Acc: {self.best_top10_score['top10_acc']}")
                print('---------------------------------------------------------------------\n')

            if self.cfg['do_logging']:
                logging_current_tensorboard(self.summary, tr_scores, ep, train=True)
                logging_current_tensorboard(self.summary, te_scores, ep, train=False)

                logging_current_txt(f"{self.cfg['log_path']}/Current_ACC_LOSS.text", te_scores, ep)

            if self.cfg['do_ckp_save']:
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "AdamW_state_dict": self.optimizer.state_dict(),
                        "epoch": ep,
                    },
                    os.path.join(f"{self.cfg['ckp_path']}/", f"{ep}.pth"),
                )

        if self.cfg['do_logging']:
            logging_best_score(f"{self.cfg['log_path']}/best_top1_score.text", self.best_top1_score)
            logging_best_score(f"{self.cfg['log_path']}/best_top5_score.text", self.best_top5_score)
            logging_best_score(f"{self.cfg['log_path']}/best_top10_score.text", self.best_top10_score)



