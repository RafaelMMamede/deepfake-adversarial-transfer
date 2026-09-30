import os
import logging
import torch
import torchvision
import torch.nn as nn
from metrics.registry import BACKBONE

logger = logging.getLogger(__name__)

@BACKBONE.register_module(module_name="resnet34")
class ResNet34(nn.Module):
    def __init__(self, resnet_config):
        super(ResNet34, self).__init__()
        """
        Args:
            resnet_config: configuration dict
        """
        self.num_classes = resnet_config["num_classes"]
        inc = resnet_config["inc"]
        self.mode = resnet_config["mode"]
        pretrained_path = resnet_config.get("pretrained", None)

        # build model without triggering online download
        resnet = torchvision.models.resnet34(pretrained=False)

        # adapt input channels if needed
        if inc != 3:
            resnet.conv1 = nn.Conv2d(
                inc, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # load local pretrained weights if provided
        if pretrained_path is not None:
            if not os.path.isfile(pretrained_path):
                raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")

            state_dict = torch.load(pretrained_path, map_location="cpu")

            # official torchvision weights are a plain state_dict
            # remove fc if num_classes differs
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}

            missing, unexpected = resnet.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded pretrained ResNet34 weights from: {pretrained_path}")
            logger.info(f"Missing keys: {missing}")
            logger.info(f"Unexpected keys: {unexpected}")

        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)

        if self.mode == 'adjust_channel':
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(512, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

    def features(self, inp):
        x = self.resnet(inp)
        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)
        return x

    def classifier(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, inp):
        x = self.features(inp)
        out = self.classifier(x)
        return out