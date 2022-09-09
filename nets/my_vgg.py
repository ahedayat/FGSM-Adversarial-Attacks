import torch.nn as nn
from torchvision.models import (vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn)

from .nets_utlis import load_net as load


class MyVGG(nn.Module):
    """
        A simple Network for classification.
    """

    def __init__(self,
                 vgg_type,
                 num_classes,
                 pretrained=True,
                 dropout=0.5,
                 freeze_backbone=False
                 ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.vgg_type = vgg_type

        assert self.vgg_type in ["vgg19", "vgg16", "vgg13",
                                 "vgg11"], "VGG type ('vgg_type') must be one of this elementes: ['vgg19', 'vgg16', 'vgg13', 'vgg11']. "

        vgg_fn = None
        if self.vgg_type == "vgg11":
            vgg_fn = vgg11_bn
        elif self.vgg_type == "vgg13":
            vgg_fn = vgg13_bn
        elif self.vgg_type == "vgg16":
            vgg_fn = vgg16_bn
        else:
            vgg_fn = vgg19_bn

        self.vgg = vgg_fn(pretrained=pretrained, progress=True)
        
        input_lastLayer = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(input_lastLayer,num_classes)
        # self.vgg.classifier = nn.Identity()

        # self.my_classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(4096, self.num_classes),
        #     nn.Softmax()
        # )

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, _in):
        """
            Forwarding input to output
        """
        out = self.vgg(_in)

        # out = self.my_classifier(out)
        # out = self.softmax(out)

        return out

    def freeze_backbone(self):
        """
            Freezing Backbone Network
        """
        for param in self.vgg.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
            Unfreezing Backbone Network
        """
        for param in self.vgg.parameters():
            param.requires_grad = True

    def freeze_classifier(self):
        """
            Freezing Classifier Network
        """
        for param in self.my_classifier.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        """
            Unfreezing Classifier Network
        """
        for param in self.my_classifier.parameters():
            param.requires_grad = True
