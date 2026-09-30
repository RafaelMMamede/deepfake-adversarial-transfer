import logging
import torch
import torch.nn as nn
import timm

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='deit_detector_custom')
class DeiTDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.backbone = self.build_backbone(config)
        feature_dim = int(getattr(self.backbone, "num_features", 384))
        self.head = self.build_classifier(config, feature_dim)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        model_name = config.get(
        "backbone_model",
        config.get("backbone_name", "deit_small_patch16_224")
    )
        pretrained = bool(config.get("pretrained", False))
        checkpoint_path = config.get("checkpoint_path", None)
        img_size = int(config.get("img_size", 224))
        in_chans = int(config.get("in_chans", 3))

        # DeiT / ViT regularization
        drop_rate = float(config.get("drop_rate", 0.0))
        attn_drop_rate = float(config.get("attn_drop_rate", 0.0))
        drop_path_rate = float(config.get("drop_path_rate", 0.0))

        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            in_chans=in_chans,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu")

            if isinstance(ckpt, dict):
                if "model" in ckpt:
                    ckpt = ckpt["model"]
                elif "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]

            cleaned = {}
            for k, v in ckpt.items():
                k = k.replace("module.", "")
                k = k.replace("backbone.", "")
                k = k.replace("model.", "")
                cleaned[k] = v

            missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
            logger.info("[DeiT] loaded checkpoint from %s", checkpoint_path)
            logger.info("[DeiT] missing keys: %s", missing)
            logger.info("[DeiT] unexpected keys: %s", unexpected)
        else:
            logger.info(
                "[DeiT] initialized model=%s pretrained=%s drop_rate=%.3f attn_drop_rate=%.3f drop_path_rate=%.3f",
                model_name,
                pretrained,
                drop_rate,
                attn_drop_rate,
                drop_path_rate,
            )

        return backbone

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

    def features(self, data_dict: dict) -> torch.Tensor:
        img = data_dict["image"]
        feat = self.backbone(img)   # [B, feature_dim]
        return feat

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]
        pred = pred_dict["cls"]
        loss = self.loss_func(pred, label)
        return {"overall": loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]
        pred = pred_dict["cls"]
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {"acc": acc, "auc": auc, "eer": eer, "ap": ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
        feat = self.features(data_dict)
        cls = self.classifier(feat)
        prob = torch.softmax(cls, dim=1)[:, 1]
        return {"cls": cls, "prob": prob, "feat": feat}

    def get_param_groups(self, config):
        opt_name = config["optimizer"]["type"]
        opt_cfg = config["optimizer"][opt_name]

        backbone_lr = opt_cfg.get("backbone_lr", opt_cfg["lr"])
        head_lr = opt_cfg.get("head_lr", opt_cfg["lr"])

        param_groups = []

        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]

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