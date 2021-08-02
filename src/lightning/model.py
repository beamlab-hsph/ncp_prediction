from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import MetricCollection, Precision, Recall, F1, Specificity, AUROC, AveragePrecision
from torch.optim import AdamW
import torchvision


class NCPModel(LightningModule):
    def __init__(self,
                 base_model: str,
                 feature_extract: bool,
                 use_pretrained: bool,
                 pos_weight: float) -> None:
        super().__init__()
        self.model_args = {
            'base_model': base_model,
            'feature_extract': feature_extract,
            'use_pretrained': use_pretrained
        }
        self.model = self._initialize_model(**self.model_args)

        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.Tensor([pos_weight]))
        else:
            self.pos_weight = None

        self.train_metrics = MetricCollection([
            Precision(),
            Recall(),
            F1(),
            Specificity(),
            AUROC(pos_label=1),
            AveragePrecision(pos_label=1)
        ])
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        self.save_hyperparameters()

    def _set_parameter_requires_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def _initialize_model(self, base_model, feature_extract, use_pretrained):
        if base_model == "densenet121":
            model = torchvision.models.densenet121(pretrained=use_pretrained)
            self._set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 1),
            )
        else:
            raise ValueError("Invalid model name")

        return model

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(
            logits, y.float(), pos_weight=self.pos_weight)
        self.train_metrics.update(logits, y)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outs):
        for metric, value in self.train_metrics.compute().items():
            self.log(f'train_{metric}_epoch', value,
                     prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.val_metrics.update(logits, y)
        loss = F.binary_cross_entropy_with_logits(
            logits, y.float(), pos_weight=self.pos_weight)
        self.log('valid_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_epoch_end(self, outs):
        for metric, value in self.val_metrics.compute().items():
            self.log(f'valid_{metric}_epoch', value,
                     prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.test_metrics.update(logits, y)
        loss = F.binary_cross_entropy_with_logits(
            logits, y.float(), pos_weight=self.pos_weight)
        self.log('test_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_epoch_end(self, outs):
        for metric, value in self.test_metrics.compute().items():
            self.log(f'test_{metric}_epoch', value,
                     prog_bar=True, logger=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)


    @staticmethod
    def add_model_specific_args(parent_parser):
        # parser = parent_parser.add_argument_group("NCPModel")
        parent_parser.add_argument('--base_model', type=str, default='densenet121')
        parent_parser.add_argument('--feature_extract', type=bool, default=False)
        parent_parser.add_argument('--use_pretrained', type=bool, default=True)
        
        return parent_parser