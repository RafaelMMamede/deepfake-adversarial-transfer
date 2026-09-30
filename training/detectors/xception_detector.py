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

@DETECTOR.register_module(module_name='xception')
class XceptionDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.video_names = []
        self.correct, self.total = 0, 0
        
    def build_backbone(self, config):
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)

        pretrained_path = config.get('pretrained', None)

        if pretrained_path is not None:
            if not os.path.isfile(pretrained_path):
                raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")

            logger.info(f"Loading pretrained Xception weights from: {pretrained_path}")

            state_dict = torch.load(pretrained_path, map_location="cpu")

            # If checkpoint is wrapped, unwrap common formats
            if isinstance(state_dict, dict):
                if "detector_backbone" in state_dict:
                    state_dict = state_dict["detector_backbone"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]

            # Remove common prefixes
            cleaned_state = {}
            for k, v in state_dict.items():
                new_k = k
                for prefix in [
                    "module.",
                    "detector_backbone.",
                    "detector.",
                    "backbone.",
                    "model.",
                    "xception.",
                ]:
                    if new_k.startswith(prefix):
                        new_k = new_k[len(prefix):]

                cleaned_state[new_k] = v

            state_dict = cleaned_state

            # Remove classifier/head keys
            state_dict = {
                k: v for k, v in state_dict.items()
                if not (
                    k.startswith("fc.")
                    or k.startswith("last_linear.")
                    or k.startswith("classifier.")
                    or k.startswith("last_layer.")
                )
            }

            # Safety: fix only if checkpoint still has bad 6D pointwise tensors
            fixed = []
            for k, v in list(state_dict.items()):
                if (
                    "pointwise.weight" in k
                    and torch.is_tensor(v)
                    and v.ndim == 6
                    and tuple(v.shape[-4:]) == (1, 1, 1, 1)
                ):
                    state_dict[k] = v.reshape(v.shape[0], v.shape[1], 1, 1)
                    fixed.append(k)

            if fixed:
                logger.info(f"Fixed {len(fixed)} 6D Xception pointwise weights before loading.")

            # Filter incompatible keys before load_state_dict
            model_state = backbone.state_dict()
            filtered_state = {}

            skipped = []
            for k, v in state_dict.items():
                if k in model_state and torch.is_tensor(v) and model_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    expected = tuple(model_state[k].shape) if k in model_state else "missing_in_model"
                    found = tuple(v.shape) if torch.is_tensor(v) else type(v)
                    skipped.append((k, found, expected))

            missing, unexpected = backbone.load_state_dict(filtered_state, strict=False)

            logger.info(f"Loaded pretrained Xception weights from: {pretrained_path}")
            logger.info(f"Loaded compatible keys: {len(filtered_state)}")
            logger.info(f"Skipped incompatible keys: {len(skipped)}")
            logger.info(f"Missing keys: {len(missing)}")
            logger.info(f"Unexpected keys: {len(unexpected)}")

            if skipped:
                logger.info("First skipped keys:")
                for k, found, expected in skipped[:20]:
                    logger.info(f"  {k}: checkpoint={found}, model={expected}")

        else:
            logger.info("No pretrained Xception weights provided. Training from scratch.")

        return backbone
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image']) #32,3,256,256

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        overall_loss = loss
        loss_dict = {'overall': overall_loss, 'cls': loss,}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        # we dont compute the video-level metrics for training
        self.video_names = []
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

        # Xception classifier layers are usually called last_linear,
        # classifier, fc, or head depending on the implementation.
        head_keywords = [
            "classifier",
            "fc",
            "last_linear",
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

        logger.info(
            f"[Xception] parameter groups: "
            f"backbone_params={sum(p.numel() for p in backbone_params)}, "
            f"head_params={sum(p.numel() for p in head_params)}, "
            f"backbone_lr={backbone_lr}, "
            f"head_lr={head_lr}"
        )

        return param_groups