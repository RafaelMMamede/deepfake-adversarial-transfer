import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='resnet34')
class ResnetDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = dict(config['backbone_config'])
        model_config['pretrained'] = config.get('pretrained', None)
        backbone = backbone_class(model_config)
        return backbone
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image'])

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict

    def get_param_groups(self, config):
        opt_name = config["optimizer"]["type"]
        opt_cfg = config["optimizer"][opt_name]

        backbone_lr = opt_cfg.get("backbone_lr", opt_cfg["lr"])
        head_lr = opt_cfg.get("head_lr", opt_cfg["lr"])

        backbone_params = []
        head_params = []

        head_keywords = [
            "classifier",
            "fc",
            "last_layer",
            "head",
        ]

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            name_lower = name.lower()

            if any(keyword in name_lower for keyword in head_keywords):
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = []

        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": backbone_lr,
            })

        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": head_lr,
            })

        return param_groups

