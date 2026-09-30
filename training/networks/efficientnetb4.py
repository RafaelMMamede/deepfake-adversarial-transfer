import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from metrics.registry import BACKBONE


@BACKBONE.register_module(module_name="efficientnetb4")
class EfficientNetB4(nn.Module):
    def __init__(self, efficientnetb4_config):
        super(EfficientNetB4, self).__init__()

        self.num_classes = efficientnetb4_config["num_classes"]
        inc = efficientnetb4_config["inc"]
        self.dropout = efficientnetb4_config["dropout"]
        self.mode = efficientnetb4_config["mode"]

        pretrained_path = efficientnetb4_config.get("pretrained", None)

        if pretrained_path is not None and pretrained_path != "None":
            if not os.path.isfile(pretrained_path):
                raise FileNotFoundError(f"EfficientNet checkpoint not found: {pretrained_path}")
            self.efficientnet = EfficientNet.from_pretrained(
                "efficientnet-b4",
                weights_path=pretrained_path,
            )
        else:
            self.efficientnet = EfficientNet.from_name("efficientnet-b4")

        # only replace stem if needed
        if inc != 3:
            self.efficientnet._conv_stem = nn.Conv2d(
                inc, 48, kernel_size=3, stride=2, bias=False
            )

        # remove original classifier
        self.efficientnet._fc = nn.Identity()

        if self.dropout:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        feat_dim = 512 if self.mode == "adjust_channel" else 1792
        self.last_layer = nn.Linear(feat_dim, self.num_classes)

        if self.mode == "adjust_channel":
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(1792, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

    def block_part1(self, x):
        x = self.efficientnet._swish(
            self.efficientnet._bn0(self.efficientnet._conv_stem(x))
        )
        for idx, block in enumerate(self.efficientnet._blocks[:10]):
            drop_connect_rate = self.efficientnet._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.efficientnet._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        return x

    def block_part2(self, x):
        for idx, block in enumerate(self.efficientnet._blocks[10:22]):
            drop_connect_rate = self.efficientnet._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx + 10) / len(self.efficientnet._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        return x

    def block_part3(self, x):
        for idx, block in enumerate(self.efficientnet._blocks[22:]):
            drop_connect_rate = self.efficientnet._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx + 22) / len(self.efficientnet._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        x = self.efficientnet._swish(
            self.efficientnet._bn1(self.efficientnet._conv_head(x))
        )
        return x

    def features(self, x):
        x = self.efficientnet.extract_features(x)
        if self.mode == "adjust_channel":
            x = self.adjust_channel(x)
        return x

    def end_points(self, x):
        return self.efficientnet.extract_endpoints(x)

    def classifier(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        if self.dropout:
            x = self.dropout_layer(x)

        self.last_emb = x
        y = self.last_layer(x)
        return y

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x