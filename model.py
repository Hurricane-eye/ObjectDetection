import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def get_model():
    backbone = torchvision.models.resnet34(pretrained=True)

    backbone.out_channels = 1000

    anchor_generator = AnchorGenerator(sizes=(()))