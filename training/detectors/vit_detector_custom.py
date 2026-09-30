import logging
from typing import Dict

import timm
import torch
import torch.nn as nn

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name="vit_detector_custom")
class ViTDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone, self.feature_dim = self.build_backbone(config)
        self.head = self.build_classifier(config, self.feature_dim)
        self.loss_func = self.build_loss(config)

        self.prob, self.label = [], []
        self.video_names = []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        backbone_cfg = config


        backbone_name = backbone_cfg.get("backbone_name", "vit_base_patch16_224")

        checkpoint_path = backbone_cfg.get("checkpoint_path", None)
        in_chans = backbone_cfg.get("in_chans", 3)
        img_size = backbone_cfg.get("img_size", 224)

        drop_rate = backbone_cfg.get("drop_rate", 0.0)
        attn_drop_rate = backbone_cfg.get("attn_drop_rate", 0.0)
        drop_path_rate = backbone_cfg.get("drop_path_rate", 0.0)

        pretrained = bool(backbone_cfg.get("pretrained", False))

        # num_classes=0 -> feature embedding output instead of classifier logits
        backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained if checkpoint_path is None else False,
            num_classes=0,
            in_chans=in_chans,
            img_size=img_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")

            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model" in state:
                state = state["model"]
            elif isinstance(state, dict) and "detector_backbone" in state:
                state = state["detector_backbone"]
            elif isinstance(state, dict) and "backbone" in state:
                state = state["backbone"]

            cleaned = {}
            for k, v in state.items():
                k = k.replace("module.", "")
                k = k.replace("backbone.", "")
                k = k.replace("detector.backbone.", "")
                cleaned[k] = v

            missing, unexpected = backbone.load_state_dict(cleaned, strict=False)

            logger.info(f"Loaded ViT checkpoint from {checkpoint_path}")
            logger.info(f"Missing keys: {missing}")
            logger.info(f"Unexpected keys: {unexpected}")

        else:
            logger.info(f"Using timm pretrained={pretrained} for {backbone_name}")

        feature_dim = backbone.num_features
        logger.info(f"ViT backbone_name = {backbone_name}")
        logger.info(f"ViT feature_dim = {feature_dim}")

        return backbone, feature_dim

    def build_classifier(self, config, feature_dim):
        hidden_dim = config.get("classifier_hidden_dim", 256)
        dropout = config.get("classifier_dropout", 0.2)

        return nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def build_loss(self, config):
        loss_class = LOSSFUNC[config["loss_func"]]
        return loss_class()

    def features(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = data_dict["image"]

        # With timm ViT and num_classes=0, this should return [B, C]
        feat = self.backbone(x)

        # Safety in case a future timm/model variant returns token features [B, N, C]
        if feat.ndim == 3:
            feat = feat[:, 0]

        return feat

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]
        pred = pred_dict["cls"]

        loss = self.loss_func(pred, label)

        return {
            "overall": loss,
            "cls": loss,
        }

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]
        pred = pred_dict["cls"]

        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(), pred.detach()
        )

        self.video_names = []

        return {
            "acc": acc,
            "auc": auc,
            "eer": eer,
            "ap": ap,
        }

    def forward(self, data_dict: dict, inference=False) -> dict:
        feat = self.features(data_dict)
        pred = self.classifier(feat)
        prob = torch.softmax(pred, dim=1)[:, 1]

        return {
            "cls": pred,
            "prob": prob,
            "feat": feat,
        }

    def get_param_groups(self, config):
        opt_name = config["optimizer"]["type"]
        opt_cfg = config["optimizer"][opt_name]

        backbone_lr = opt_cfg.get("backbone_lr", opt_cfg["lr"])
        head_lr = opt_cfg.get("head_lr", opt_cfg["lr"])

        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]

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