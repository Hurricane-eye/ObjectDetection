import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
import utils
from engine import train_one_epoch, evaluate
from VisDrone import *


def get_model():
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 12  # 12 classes (person + car) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-train head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # num_classes = 12
    # # load a pre-trained model for classification and return
    # # only the features
    # backbone = torchvision.models.resnet34(pretrained=True).features
    # # FasterRCNN needs to know the number of
    # # output channels in a backbone. For mobilenet_v2, it's 1280
    # # so we need to add it here
    # backbone.out_channels = 1280
    #
    # # let's make the RPN generate 5x3 anchors per spatial
    # # location, with 5 different sizes and 3 different aspect
    # # ratios. We have a Tuple[Tuple[int]] because each feature
    # # map could potentially have different sizes and aspect ratios
    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                    aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # # let's define what are the features maps that we will
    # # use to perform the region of interest cropping, as well as
    # # the size of the crop after rescaling.
    # # if your backbone returns a Tensor, featmap_names is expected to
    # # be [0]. More generally, the backbone should return an
    # # OrderedDict[Tensor], and in featmap_names you can choose which
    # # feature maps to use
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
    #                                                 output_size=7, sampling_ratio=2)
    # # put the pieces together inside a FasterRCNN model
    # model = FasterRCNN(backbone, num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    return model




def main():
    model = get_model()

    # use our dataset
    train_set = VisDroneDataset("VisDrone2019-DET-train", get_transform(train=True))
    data_loader = DataLoader(dataset=train_set, batch_size=2,
                             shuffle=True, num_workers=4,
                             collate_fn=utils.collate_fn)
    # define validation data loader
    val_set = VisDroneDataset("VisDrone2019-DET-val", get_transform(train=False))
    data_loader_val = DataLoader(dataset=val_set, batch_size=2,
                                 shuffle=False, num_workers=4,
                                 collate_fn=utils.collate_fn)
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # construct an optimizer
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rete scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    num_epochs = 10
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the val dataset
        evaluate(model, data_loader_val, device=device)
    torch.save(model.state_dict(), "fasterrcnn_resnet50.pt")


if __name__ == '__main__':
    main()
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5x3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the features maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7, sampling_ratio=2)
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone, num_classes=3,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
