import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_model(pretrained=True):
    # # load a model pre-trained on COCO
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrain, min_size=540, max_size=960)
    # # replace the classifier with a new one, that has
    # # num_classes which is user-defined
    # num_classes = 13
    # # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-train head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    backbone = resnet_fpn_backbone("resnet50", pretrained)
    # backbone.out_channels = 256
    # anchor_size =((8,), (16,), (32,), (64,), (128,), (256,), (512,))
    anchor_size = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_size)
    rpn_anchor_generator = AnchorGenerator(anchor_size, aspect_ratios)
    model = FasterRCNN(backbone=backbone,
                       num_classes=13,
                       rpn_anchor_generator=rpn_anchor_generator,
                       min_size=540,
                       max_size=900)

    return model